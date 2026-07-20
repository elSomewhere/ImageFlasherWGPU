// ============================================================================
// Monochrome data-materialization shaders for ImageFlasherWGPU
//
// Three fragment stages share one 96-byte RenderParams uniform:
//
//   1. Tile pass   (ikedaImageFlasherFragmentWGSL)
//      Every tile renders its ring-buffer image through a per-tile "material":
//      raw luma, 1-bit threshold, Bayer halftone, Sobel wireframe, pixel-sort
//      smear, slice glitch, waveform readout, barcode collapse, hex-glyph
//      rain, or mosaic blocks. Tiles choose between two active styles by
//      stable per-tile hash, so the wall is heterogeneous but coherent.
//
//   2. Mosh pass   (ikedaFadeFragmentWGSL)
//      Temporal feedback. Blends the previous frame into the new one, but the
//      previous frame is re-sampled through block displacement (broken motion
//      vectors) and per-block "P-frame drops" that hold stale image data.
//      This is the datamoshing engine; fade remains the base blend factor.
//
//   3. Present pass (ikedaPresentFragmentWGSL)
//      Global composition: scroll, slice glitches on event pulses, scanlines,
//      rolling sync bar, hairline grid, bit-noise, strobe/invert, binary
//      timecode strip, hard monochrome enforcement.
//
// All texture reads use textureSampleLevel so styles may branch on per-tile
// (non-uniform) values without violating WGSL uniformity rules.
// ============================================================================

#include <string>

// Shared uniform block + helpers, injected into each fragment shader below.
static const char* renderParamsWGSL = R"(
struct RenderParams {
    time : f32,          // seconds
    eventPulse : f32,    // 0..1 impulse, decays in C++
    strobe : f32,        // 0..1 strobe intensity
    bypass : f32,        // 0..1 neutral blend: 1 = raw images, no treatment

    styleA : f32,        // primary tile material id
    styleB : f32,        // secondary tile material id
    styleMixProb : f32,  // probability a tile uses styleB
    threshold : f32,     // 1-bit threshold center

    ditherScale : f32,   // Bayer cell size in pixels
    blockScale : f32,    // glyph/mosaic cell size in pixels
    sliceAmp : f32,      // slice displacement amplitude (uv)
    jitterAmp : f32,     // per-tile threshold jitter

    moshAmount : f32,    // block displacement strength
    moshBlock : f32,     // mosh block size (uv)
    moshDrop : f32,      // probability a block holds stale data
    feedbackDecay : f32, // luminance decay of held blocks

    invert : f32,        // 0..1 global inversion
    gridOverlay : f32,   // hairline grid + timecode intensity
    scanline : f32,      // scanline + rolling bar intensity
    noiseAmount : f32,   // bit-flip noise intensity

    canvasWidth : f32,
    canvasHeight : f32,
    colorBleed : f32,    // 0 = strict mono, 1 = source color
    contrast : f32,      // luma contrast around 0.5

    flashBoost : f32,    // white flash strength on fresh tile switches
    tileGutter : f32,    // black gutter between tiles (contact-sheet look)
    sequenceLow : f32,   // low 16 bits of the latest artifact sequence
    reserved0 : f32
}

fn luma(c : vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn hash11(p : f32) -> f32 {
    var x = fract(p * 0.1031);
    x = x * (x + 33.33);
    x = x * (x + x);
    return fract(x);
}

fn hash21(p : vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn bayer4(pos : vec2<f32>) -> f32 {
    var m = array<f32, 16>(
         0.0,  8.0,  2.0, 10.0,
        12.0,  4.0, 14.0,  6.0,
         3.0, 11.0,  1.0,  9.0,
        15.0,  7.0, 13.0,  5.0
    );
    let x = u32(pos.x) % 4u;
    let y = u32(pos.y) % 4u;
    return (m[y * 4u + x] + 0.5) / 16.0;
}

// 4x5 hex glyphs, one packed u32 per digit; bit index = row*4 + col, col 0 left.
fn hexGlyph(v : u32, cell : vec2<u32>) -> f32 {
    var glyphs = array<u32, 16>(
        (6u|(9u<<4u)|(9u<<8u)|(9u<<12u)|(6u<<16u)),   // 0
        (2u|(3u<<4u)|(2u<<8u)|(2u<<12u)|(7u<<16u)),   // 1
        (6u|(9u<<4u)|(4u<<8u)|(2u<<12u)|(15u<<16u)),  // 2
        (7u|(8u<<4u)|(6u<<8u)|(8u<<12u)|(7u<<16u)),   // 3
        (9u|(9u<<4u)|(15u<<8u)|(8u<<12u)|(8u<<16u)),  // 4
        (15u|(1u<<4u)|(7u<<8u)|(8u<<12u)|(7u<<16u)),  // 5
        (6u|(1u<<4u)|(7u<<8u)|(9u<<12u)|(6u<<16u)),   // 6
        (15u|(8u<<4u)|(4u<<8u)|(2u<<12u)|(2u<<16u)),  // 7
        (6u|(9u<<4u)|(6u<<8u)|(9u<<12u)|(6u<<16u)),   // 8
        (6u|(9u<<4u)|(14u<<8u)|(8u<<12u)|(6u<<16u)),  // 9
        (6u|(9u<<4u)|(15u<<8u)|(9u<<12u)|(9u<<16u)),  // A
        (7u|(9u<<4u)|(7u<<8u)|(9u<<12u)|(7u<<16u)),   // B
        (6u|(9u<<4u)|(1u<<8u)|(9u<<12u)|(6u<<16u)),   // C
        (7u|(9u<<4u)|(9u<<8u)|(9u<<12u)|(7u<<16u)),   // D
        (15u|(1u<<4u)|(7u<<8u)|(1u<<12u)|(15u<<16u)), // E
        (15u|(1u<<4u)|(7u<<8u)|(1u<<12u)|(1u<<16u))   // F
    );
    if (cell.x >= 4u || cell.y >= 5u) { return 0.0; }
    let bit = (glyphs[v & 15u] >> (cell.y * 4u + cell.x)) & 1u;
    return f32(bit);
}
)";

// ==================== TILE PASS ====================
// Renders one instanced quad per tile. Style ids:
//   0 RAW  1 THRESH  2 BAYER  3 EDGE  4 SORT  5 SLICE
//   6 WAVE 7 BARCODE 8 HEX    9 BLOCKS
static const std::string ikedaImageFlasherFragmentSrc = std::string(renderParamsWGSL) + R"(
@group(0) @binding(1) var texArr : texture_2d_array<f32>;
@group(0) @binding(2) var samp : sampler;
@group(0) @binding(3) var<uniform> P : RenderParams;

fn tileLuma(uv : vec2<f32>, layer : i32) -> f32 {
    let c = textureSampleLevel(texArr, samp, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), layer, 0.0);
    return luma(c.rgb);
}

fn shapeLuma(g : f32) -> f32 {
    return clamp((g - 0.5) * max(P.contrast, 0.01) + 0.5, 0.0, 1.0);
}

@fragment
fn fsImage(
    @builtin(position) fragPos : vec4<f32>,
    @location(0) uv : vec2<f32>,
    @location(1) @interpolate(flat) layerIndex : i32,
    @location(2) @interpolate(flat) instanceId : u32,
    @location(3) @interpolate(flat) tileAge : f32
) -> @location(0) vec4<f32> {
    let inst = f32(instanceId);
    let tileSeed = hash11(inst * 17.13 + 0.37);          // stable per tile
    let flashSeed = hash11(inst * 7.77 + f32(layerIndex) * 3.71); // re-rolls on switch

    // Choose material per tile.
    var style = i32(P.styleA + 0.5);
    if (hash11(inst * 5.19 + 11.7) < P.styleMixProb) {
        style = i32(P.styleB + 0.5);
    }

    let srcColor = textureSampleLevel(texArr, samp, uv, layerIndex, 0.0);
    var g = shapeLuma(luma(srcColor.rgb));
    var outv = g;

    switch style {
        case 0: { // RAW grayscale
            outv = g;
        }
        case 1: { // THRESH: 1-bit, per-tile jittered threshold
            let t = clamp(P.threshold + (tileSeed - 0.5) * P.jitterAmp, 0.05, 0.95);
            outv = step(t, g);
        }
        case 2: { // BAYER halftone
            let cellPx = max(P.ditherScale, 1.0);
            outv = step(bayer4(floor(fragPos.xy / cellPx)), g);
        }
        case 3: { // EDGE: Sobel wireframe on black
            let e = 1.0 / 384.0;
            let l  = tileLuma(uv + vec2<f32>(-e, 0.0), layerIndex);
            let r  = tileLuma(uv + vec2<f32>( e, 0.0), layerIndex);
            let u2 = tileLuma(uv + vec2<f32>(0.0, -e), layerIndex);
            let d  = tileLuma(uv + vec2<f32>(0.0,  e), layerIndex);
            let mag = length(vec2<f32>(r - l, d - u2)) * 4.0;
            outv = step(P.threshold * 0.6, mag);
        }
        case 4: { // SORT: luma-keyed horizontal smear (pixel-sort impression)
            let bands = 48.0;
            let band = floor(uv.y * bands) / bands;
            let key = tileLuma(vec2<f32>(0.03, band + 0.5 / bands), layerIndex);
            var acc = 0.0;
            let reach = (0.02 + 0.25 * key) * (0.5 + tileSeed);
            for (var i = 0; i < 8; i = i + 1) {
                let s = tileLuma(vec2<f32>(uv.x - reach * f32(i) / 8.0, uv.y), layerIndex);
                acc = max(acc, s);
            }
            outv = step(P.threshold, shapeLuma(acc));
        }
        case 5: { // SLICE: displaced horizontal bands + posterize
            let sliceH = 0.04 + 0.08 * tileSeed;
            let band = floor(uv.y / sliceH);
            let roll = floor(P.time * 6.0);
            let off = (hash21(vec2<f32>(band, roll + inst)) - 0.5)
                      * P.sliceAmp * step(0.6, hash21(vec2<f32>(band + 7.0, roll)));
            let s = tileLuma(vec2<f32>(uv.x + off, uv.y), layerIndex);
            outv = floor(shapeLuma(s) * 4.0) / 3.0;
        }
        case 6: { // WAVE: image rows re-drawn as waveform columns
            let cols = 96.0;
            let xq = (floor(uv.x * cols) + 0.5) / cols;
            let h = tileLuma(vec2<f32>(xq, 0.25 + 0.5 * flashSeed), layerIndex);
            let bar = step(1.0 - uv.y, h);
            let axis = step(abs(uv.y - 0.5), 0.004);
            outv = max(bar * step(P.threshold * 0.5, h), axis);
        }
        case 7: { // BARCODE: columns collapsed to vertical stripes
            let cols = 160.0;
            let xq = (floor(uv.x * cols) + 0.5) / cols;
            let v = tileLuma(vec2<f32>(xq, 0.5), layerIndex);
            outv = step(0.5, fract(v * 9.73 + xq * 3.0));
        }
        case 8: { // HEX: image blocks as hex digits
            let n = clamp(floor(384.0 / max(P.blockScale, 6.0)), 8.0, 64.0);
            let cell = floor(uv * n);
            let cellUV = fract(uv * n);
            let v = tileLuma((cell + 0.5) / n, layerIndex);
            let digit = u32(clamp(v * 15.99, 0.0, 15.0));
            // glyph occupies central 4x5 of a 6x7 cell
            let gpos = vec2<u32>(
                u32(clamp(floor(cellUV.x * 6.0) - 1.0, 0.0, 5.0)),
                u32(clamp(floor(cellUV.y * 7.0) - 1.0, 0.0, 6.0))
            );
            let on = hexGlyph(digit, gpos);
            // dark cells print dim digits, bright cells bright digits
            outv = on * (0.25 + 0.75 * step(0.35, v));
        }
        case 9: { // BLOCKS: hard mosaic with dropout
            let n = 24.0 + floor(tileSeed * 24.0);
            let cell = floor(uv * n);
            let cuv = (cell + 0.5) / n;
            let v = shapeLuma(tileLuma(cuv, layerIndex));
            let drop = step(0.92, hash21(cell + floor(P.time * 3.0) * 0.31 + inst));
            outv = floor(v * 3.0) / 2.0 * (1.0 - drop);
        }
        case 10: { // BITPLANE: single extracted bit-plane of luma
            let plane = (u32(tileSeed * 7.99) + u32(P.time * 0.4)) % 8u;
            let bits = u32(clamp(g, 0.0, 1.0) * 255.0);
            outv = f32((bits >> plane) & 1u);
        }
        case 11: { // CONTOUR: quantized luma iso-lines
            let levels = 7.0;
            let q = g * levels;
            let d = min(fract(q), 1.0 - fract(q));
            let line = 1.0 - step(0.09, d);
            let fill = floor(q) / levels * 0.18; // faint terraced fill under the lines
            outv = max(line, fill);
        }
        default: {
            outv = g;
        }
    }

    let live = 1.0 - clamp(P.bypass, 0.0, 1.0);

    // Occasional per-tile negative on fresh switches, driven by event energy.
    let neg = step(0.85, flashSeed) * clamp(P.eventPulse * 2.0, 0.0, 1.0) * live;
    outv = mix(outv, 1.0 - outv, neg);

    // Fresh-switch white flash: the moment of arrival made visible.
    let flash = P.flashBoost * exp(-tileAge / 2.5) * live;
    outv = mix(outv, 1.0, clamp(flash, 0.0, 1.0));

    let mono = vec3<f32>(outv);
    var outc = mix(mono, srcColor.rgb * outv, P.colorBleed);

    // Neutral blend: pull toward the raw image as collected.
    outc = mix(outc, srcColor.rgb, clamp(P.bypass, 0.0, 1.0));

    // Ring-slot stamp: 8-bit binary strip along the tile's bottom edge.
    if (P.gridOverlay > 0.001 && uv.y > 0.94 && uv.y < 0.985 && uv.x > 0.06 && uv.x < 0.94) {
        let cells = 8.0;
        let cx = (uv.x - 0.06) / 0.88;
        let idx = u32(clamp(floor(cx * cells), 0.0, cells - 1.0));
        let bit = (u32(layerIndex) >> idx) & 1u;
        let inCell = step(0.2, fract(cx * cells)) * step(fract(cx * cells), 0.8);
        outc = mix(outc, vec3<f32>(f32(bit)), P.gridOverlay * inCell * 0.85);
    }

    // Contact-sheet gutter between tiles.
    let e = P.tileGutter * 0.03;
    if (e > 0.0001) {
        let border = step(uv.x, e) + step(1.0 - e, uv.x) + step(uv.y, e) + step(1.0 - e, uv.y);
        outc = outc * (1.0 - clamp(border, 0.0, 1.0));
    }

    return vec4<f32>(outc, 1.0);
}
)";
const char* ikedaImageFlasherFragmentWGSL = ikedaImageFlasherFragmentSrc.c_str();

// ==================== MOSH PASS (temporal feedback) ====================
static const std::string ikedaFadeFragmentSrc = std::string(renderParamsWGSL) + R"(
@group(0) @binding(0) var oldFrame : texture_2d<f32>;
@group(0) @binding(1) var newFrame : texture_2d<f32>;

struct FadeParams {
    fade : f32
}

@group(0) @binding(2) var<uniform> fadeParam : FadeParams;
@group(0) @binding(3) var s : sampler;
@group(0) @binding(4) var<uniform> P : RenderParams;

@fragment
fn fsFade(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    let live = 1.0 - clamp(P.bypass, 0.0, 1.0);
    let bs = max(P.moshBlock, 0.004);
    let block = floor(uv / bs);
    let roll = floor(P.time * 7.0); // re-roll displacement 7x/s

    // Event pulses spike the mosh hard: the crawl "hitting" the frame.
    let amt = P.moshAmount * (1.0 + P.eventPulse * 5.0) * live;

    // Broken motion vectors: some blocks fetch the old frame from elsewhere.
    var moshUV = uv;
    let r = hash21(block + roll * 0.117);
    let r2 = hash21(block * 4.7 + roll * 0.31);
    if (r < 0.5 * clamp(amt * 4.0, 0.0, 1.0)) {
        if (r2 < 0.45) {
            // continuous smear: the block slides sideways, P-frame melt style
            let dir = sign(r2 - 0.22);
            moshUV.x = uv.x - dir * amt * 0.22 * fract(P.time * (0.25 + r2 * 0.9));
        } else {
            // rectangular jump, horizontally biased
            let dir = vec2<f32>(
                hash21(block * 1.7 + roll) - 0.5,
                (hash21(block * 2.3 + roll) - 0.5) * 0.3
            );
            moshUV = uv + dir * amt * bs * 24.0;
        }
    }

    let cOld = textureSampleLevel(oldFrame, s, clamp(moshUV, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
    let cNew = textureSampleLevel(newFrame, s, uv, 0.0);

    // P-frame drop: a block refuses new data and decays on stale content.
    let hold = step(hash21(block * 3.1 + roll * 0.71), P.moshDrop * live);
    let f = fadeParam.fade * (1.0 - hold);

    let held = cOld * mix(P.feedbackDecay, 1.0, clamp(P.bypass, 0.0, 1.0));
    return vec4<f32>(mix(held.rgb, cNew.rgb, f), 1.0);
}
)";
const char* ikedaFadeFragmentWGSL = ikedaFadeFragmentSrc.c_str();

// ==================== PRESENT PASS (global composition) ====================
static const std::string ikedaPresentFragmentSrc = std::string(renderParamsWGSL) + R"(
@group(0) @binding(0) var oldFrame : texture_2d<f32>;
@group(0) @binding(1) var s : sampler;

struct ScrollParams {
    offset : vec2<f32>
}

@group(0) @binding(2) var<uniform> scrollParam : ScrollParams;
@group(0) @binding(3) var<uniform> P : RenderParams;

@fragment
fn fsPresent(
    @builtin(position) fragPos : vec4<f32>,
    @location(0) uv : vec2<f32>
) -> @location(0) vec4<f32> {
    var suv = fract(uv + scrollParam.offset);
    let live = 1.0 - clamp(P.bypass, 0.0, 1.0);

    // Event-driven slice glitch: horizontal strips shear on pulses.
    let pulse = clamp(P.eventPulse, 0.0, 1.0) * live;
    if (pulse > 0.01) {
        let band = floor(suv.y * 28.0);
        let roll = floor(P.time * 18.0);
        let sel = step(0.75, hash21(vec2<f32>(band, roll)));
        let off = (hash21(vec2<f32>(band + 3.0, roll)) - 0.5) * 0.25 * pulse * sel;
        suv.x = fract(suv.x + off);
    }

    let base = textureSampleLevel(oldFrame, s, suv, 0.0);
    var g = luma(base.rgb);

    // Scanlines + rolling sync bar.
    let scan = P.scanline * live;
    if (scan > 0.001) {
        let line = 0.5 + 0.5 * sin(fragPos.y * 3.14159);
        g = g * (1.0 - scan * 0.35 * line);
        let barPos = fract(P.time * 0.11);
        let dy = abs(uv.y - barPos);
        g = g + scan * 0.5 * exp(-dy * 220.0);       // bright leading edge
        g = g * (1.0 - scan * 0.5 * exp(-dy * 40.0) * step(uv.y, barPos)); // dark wake
    }

    // Bit noise: sparse hard white/black flips.
    let noiseAmt = P.noiseAmount * live;
    if (noiseAmt > 0.001) {
        let n = hash21(fragPos.xy + fract(P.time * 977.0) * 100.0);
        let flip = step(1.0 - noiseAmt * 0.12, n);
        let val = step(0.5, hash21(fragPos.yx + P.time));
        g = mix(g, val, flip);
    }

    // Hairline grid.
    let gridAmt = P.gridOverlay * live;
    if (gridAmt > 0.001) {
        let cell = 96.0;
        let gx = step(fract(fragPos.x / cell), 1.0 / cell);
        let gy = step(fract(fragPos.y / cell), 1.0 / cell);
        g = g + (gx + gy) * 0.10 * gridAmt;
    }

    // Binary strip along the bottom edge: left half counts the clock,
    // right half counts the crawl (latest artifact sequence).
    if (gridAmt > 0.001 && uv.y > 1.0 - 12.0 / max(P.canvasHeight, 1.0)) {
        var value = u32(P.time * 100.0) & 0xFFFFu;
        var cx = uv.x * 2.0;
        if (uv.x >= 0.5) {
            value = u32(P.sequenceLow) & 0xFFFFu;
            cx = (uv.x - 0.5) * 2.0;
        }
        let bits = 16.0;
        let idx = u32(clamp(floor(cx * bits), 0.0, bits - 1.0));
        let bit = (value >> idx) & 1u;
        let inCell = step(0.15, fract(cx * bits)) * step(fract(cx * bits), 0.85);
        g = mix(g, f32(bit), gridAmt * inCell * 0.9);
    }

    // Strobe: hard photic flicker, gated so it stays rhythmic not constant.
    var inv = clamp(P.invert, 0.0, 1.0) * live;
    let strobeAmt = P.strobe * live;
    if (strobeAmt > 0.001) {
        let gate = step(0.5, fract(P.time * 9.0)) * step(fract(P.time * 0.618), strobeAmt);
        inv = clamp(inv + gate, 0.0, 1.0);
    }
    g = mix(g, 1.0 - g, inv);

    let mono = vec3<f32>(clamp(g, 0.0, 1.0));
    let outc = mix(mono, base.rgb, max(P.colorBleed * 0.6, clamp(P.bypass, 0.0, 1.0)));
    return vec4<f32>(outc, 1.0);
}
)";
const char* ikedaPresentFragmentWGSL = ikedaPresentFragmentSrc.c_str();
