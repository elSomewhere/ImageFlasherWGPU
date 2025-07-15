// Enhanced Ikeda-Inspired Shaders for ImageFlasherWGPU
// Extended Phase 2: Advanced Data Visualization Modes
// 
// New Modes Added:
// - Mode 5: FREQUENCY - Spectral analysis and frequency domain visualization
// - Mode 6: SCAN - Progressive scanning patterns inspired by "superposition"
// - Mode 7: MATRIX - Mathematical matrix operations and transformations
// - Mode 8: PULSE - Temporal rhythm and pulse patterns
// - Mode 9: NOISE - Random data generation and pattern analysis
// - Mode 10: STRIP - Horizontal/vertical strip decomposition
// - Mode 11: PHASE - Phase relationships and interference patterns
// - Mode 12: QUANTUM - Quantized levels and discrete data states

// ==================== ENHANCED IKEDA MODE UNIFORMS ====================

const char* ikedaModeUniformWGSL = R"(
struct IkedaModeParams {
    mode : i32,              // 0-12: visual modes
    threshold : f32,         // black/white threshold
    gridSize : f32,          // grid quantization size
    dataIntensity : f32,     // data overlay intensity
    time : f32,              // global time for animations
    canvasWidth : f32,       // for precise calculations
    canvasHeight : f32,      // for precise calculations
    
    // New extended parameters
    frequency : f32,         // frequency analysis parameter
    phaseShift : f32,        // phase shift for wave patterns
    noiseLevel : f32,        // noise generation level
    stripWidth : f32,        // strip decomposition width
    quantumLevels : f32,     // quantum state levels
    scanSpeed : f32,         // scanning speed
    matrixScale : f32,       // matrix transformation scale
    pulseRate : f32          // pulse rhythm rate
}
)";

// ==================== ENHANCED IMAGE FLASHER SHADER ====================

const char* ikedaImageFlasherFragmentWGSL = R"(
struct Uniforms {
    layerIndex : i32
}

struct IkedaModeParams {
    mode : i32,
    threshold : f32,
    gridSize : f32,
    dataIntensity : f32,
    time : f32,
    canvasWidth : f32,
    canvasHeight : f32,
    
    // Extended parameters
    frequency : f32,
    phaseShift : f32,
    noiseLevel : f32,
    stripWidth : f32,
    quantumLevels : f32,
    scanSpeed : f32,
    matrixScale : f32,
    pulseRate : f32
}

@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var texArr : texture_2d_array<f32>;
@group(0) @binding(2) var samp : sampler;
@group(0) @binding(3) var<uniform> ikeda : IkedaModeParams;

// Precise luminance calculation
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

// Main fragment function with simplified modes
@fragment
fn fsImage(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    var sampledColor = textureSample(texArr, samp, uv, u.layerIndex);
    
    // Mode 0: Normal (original behavior)
    if (ikeda.mode == 0) {
        return sampledColor;
    }
    
    // Mode 1: Pure Black & White with dynamic threshold
    if (ikeda.mode == 1) {
        let lum = luminance(sampledColor.rgb);
        let dynamicThreshold = ikeda.threshold + sin(ikeda.time * 1.5) * 0.05;
        let blackWhite = step(dynamicThreshold, lum);
        return vec4<f32>(blackWhite, blackWhite, blackWhite, sampledColor.a);
    }
    
    // Mode 2: Grid Quantization
    if (ikeda.mode == 2) {
        let pixelSize = 1.0 / ikeda.gridSize;
        let quantizedUV = floor(uv / pixelSize) * pixelSize + pixelSize * 0.5;
        let quantizedColor = textureSample(texArr, samp, quantizedUV, u.layerIndex);
        let lum = luminance(quantizedColor.rgb);
        let blackWhite = step(ikeda.threshold, lum);
        return vec4<f32>(blackWhite, blackWhite, blackWhite, sampledColor.a);
    }
    
    // Mode 3: Data Visualization Overlay
    if (ikeda.mode == 3) {
        let lum = luminance(sampledColor.rgb);
        let blackWhite = step(ikeda.threshold, lum);
        
        // Simple data pattern
        let barcodeCoord = uv.x * 80.0 + lum * 5.0;
        let barcodePattern = step(0.5, fract(barcodeCoord + sin(ikeda.time * 0.5) * 0.1));
        
        let blendedData = barcodePattern * ikeda.dataIntensity;
        let finalValue = clamp(blackWhite + blendedData * (1.0 - blackWhite), 0.0, 1.0);
        return vec4<f32>(finalValue, finalValue, finalValue, sampledColor.a);
    }
    
    // Mode 4: Binary Data Stream (simplified without bit operations)
    if (ikeda.mode == 4) {
        let scanlineCount = 48.0;
        let scanlineY = floor(uv.y * scanlineCount);
        let scanlineUV = vec2<f32>(uv.x, (scanlineY + 0.5) / scanlineCount);
        let scanlineColor = textureSample(texArr, samp, scanlineUV, u.layerIndex);
        let lum = luminance(scanlineColor.rgb);
        
        // Simplified binary pattern without bit operations
        let binaryPattern = step(0.5, fract(uv.x * 64.0 + lum * 8.0));
        return vec4<f32>(binaryPattern, binaryPattern, binaryPattern, sampledColor.a);
    }
    
    // Mode 5: Frequency Analysis
    if (ikeda.mode == 5) {
        let lum = luminance(sampledColor.rgb);
        let blackWhite = step(ikeda.threshold, lum);
        
        // Frequency bars based on luminance and frequency parameter
        let freqScale = ikeda.frequency * 2.0;
        let freqPattern = sin(uv.y * freqScale * 20.0 + ikeda.time * 2.0) * 0.5 + 0.5;
        let horizontalBars = step(0.6, freqPattern);
        
        // Spectral analysis overlay
        let spectralCoord = uv.x * freqScale * 10.0 + lum * 5.0;
        let spectralPattern = step(0.5, fract(spectralCoord + sin(ikeda.time) * 0.2));
        
        let combined = clamp(blackWhite + (horizontalBars + spectralPattern) * ikeda.dataIntensity * 0.3, 0.0, 1.0);
        return vec4<f32>(combined, combined, combined, sampledColor.a);
    }
    
    // Mode 6: Scan Lines
    if (ikeda.mode == 6) {
        let lum = luminance(sampledColor.rgb);
        let blackWhite = step(ikeda.threshold, lum);
        
        // Progressive scanning based on scanSpeed
        let scanPosition = fract(ikeda.time * ikeda.scanSpeed);
        let scanLine = abs(uv.y - scanPosition);
        let scanEffect = step(scanLine, 0.02);
        
        // Interlaced pattern
        let interlace = step(0.5, fract(uv.y * 240.0));
        
        let combined = clamp(blackWhite + (scanEffect + interlace * 0.3) * ikeda.dataIntensity, 0.0, 1.0);
        return vec4<f32>(combined, combined, combined, sampledColor.a);
    }
    
    // Mode 7: Matrix Transformations
    if (ikeda.mode == 7) {
        let lum = luminance(sampledColor.rgb);
        let blackWhite = step(ikeda.threshold, lum);
        
        // Rotating grid based on matrixScale
        let scale = ikeda.matrixScale;
        let rotation = ikeda.time * 0.5;
        let rotatedUV = vec2<f32>(
            uv.x * cos(rotation) - uv.y * sin(rotation),
            uv.x * sin(rotation) + uv.y * cos(rotation)
        );
        
        let gridPattern = step(0.9, fract(rotatedUV.x * scale * 20.0)) + 
                         step(0.9, fract(rotatedUV.y * scale * 20.0));
        
        let combined = clamp(blackWhite + gridPattern * ikeda.dataIntensity, 0.0, 1.0);
        return vec4<f32>(combined, combined, combined, sampledColor.a);
    }
    
    // Mode 8: Pulse Patterns
    if (ikeda.mode == 8) {
        let lum = luminance(sampledColor.rgb);
        let blackWhite = step(ikeda.threshold, lum);
        
        // Radial pulse based on pulseRate
        let center = vec2<f32>(0.5, 0.5);
        let dist = distance(uv, center);
        let pulse = sin(dist * 20.0 - ikeda.time * ikeda.pulseRate * 4.0) * 0.5 + 0.5;
        let pulsePattern = step(0.7, pulse);
        
        // Rhythmic grid
        let rhythmicGrid = step(0.8, fract(uv.x * 10.0 + sin(ikeda.time * ikeda.pulseRate) * 2.0)) +
                          step(0.8, fract(uv.y * 10.0 + cos(ikeda.time * ikeda.pulseRate) * 2.0));
        
        let combined = clamp(blackWhite + (pulsePattern + rhythmicGrid * 0.3) * ikeda.dataIntensity, 0.0, 1.0);
        return vec4<f32>(combined, combined, combined, sampledColor.a);
    }
    
    // Mode 9: Noise Patterns
    if (ikeda.mode == 9) {
        let lum = luminance(sampledColor.rgb);
        let blackWhite = step(ikeda.threshold, lum);
        
        // Structured noise based on noiseLevel
        let noiseCoord = uv * 50.0 + ikeda.time * 0.1;
        let noise1 = fract(sin(dot(noiseCoord, vec2<f32>(12.9898, 78.233))) * 43758.5453);
        let noise2 = fract(sin(dot(noiseCoord + vec2<f32>(1.0, 1.0), vec2<f32>(12.9898, 78.233))) * 43758.5453);
        
        let noisePattern = step(1.0 - ikeda.noiseLevel, noise1);
        let structuredNoise = step(0.5, noise2) * ikeda.noiseLevel;
        
        let combined = clamp(blackWhite + (noisePattern + structuredNoise) * ikeda.dataIntensity, 0.0, 1.0);
        return vec4<f32>(combined, combined, combined, sampledColor.a);
    }
    
    // Mode 10: Strip Decomposition
    if (ikeda.mode == 10) {
        let lum = luminance(sampledColor.rgb);
        let blackWhite = step(ikeda.threshold, lum);
        
        // Alternating horizontal/vertical strips based on stripWidth
        let stripSize = ikeda.stripWidth * 10.0;
        let horizontalStrips = step(0.5, fract(uv.y / stripSize));
        let verticalStrips = step(0.5, fract(uv.x / stripSize));
        
        // Time-based switching between horizontal and vertical
        let timeSwitch = step(0.5, fract(ikeda.time * 0.3));
        let stripPattern = mix(horizontalStrips, verticalStrips, timeSwitch);
        
        let combined = clamp(blackWhite * stripPattern + (1.0 - stripPattern) * blackWhite * 0.3, 0.0, 1.0);
        return vec4<f32>(combined, combined, combined, sampledColor.a);
    }
    
    // Mode 11: Phase Interference
    if (ikeda.mode == 11) {
        let lum = luminance(sampledColor.rgb);
        let blackWhite = step(ikeda.threshold, lum);
        
        // Multiple wave phases with interference
        let wave1 = sin(uv.x * 30.0 + ikeda.time + ikeda.phaseShift);
        let wave2 = sin(uv.y * 30.0 + ikeda.time * 1.2);
        let wave3 = sin((uv.x + uv.y) * 20.0 + ikeda.time * 0.8 + ikeda.phaseShift * 2.0);
        
        let interference = (wave1 + wave2 + wave3) / 3.0;
        let phasePattern = step(0.3, interference * 0.5 + 0.5);
        
        let combined = clamp(blackWhite + phasePattern * ikeda.dataIntensity * 0.4, 0.0, 1.0);
        return vec4<f32>(combined, combined, combined, sampledColor.a);
    }
    
    // Mode 12: Quantum Levels
    if (ikeda.mode == 12) {
        let lum = luminance(sampledColor.rgb);
        
        // Quantize luminance to discrete levels
        let levels = ikeda.quantumLevels;
        let quantizedLum = floor(lum * levels) / levels;
        
        // Quantum tunneling effect
        let tunnelCoord = uv * 20.0 + ikeda.time * 0.2;
        let tunnel = fract(sin(dot(tunnelCoord, vec2<f32>(12.9898, 78.233))) * 43758.5453);
        let tunnelPattern = step(0.9, tunnel) * (1.0 / levels);
        
        // Energy level visualization
        let energyLevel = floor(quantizedLum * levels) / levels;
        let levelPattern = step(ikeda.threshold, energyLevel + tunnelPattern);
        
        return vec4<f32>(levelPattern, levelPattern, levelPattern, sampledColor.a);
    }

    
    return sampledColor;
}
)";

// ==================== ENHANCED FADE SHADER ====================

const char* ikedaFadeFragmentWGSL = R"(
@group(0) @binding(0) var oldFrame : texture_2d<f32>;
@group(0) @binding(1) var newFrame : texture_2d<f32>;

struct FadeParams {
    fade : f32
}

struct IkedaModeParams {
    mode : i32,
    threshold : f32,
    gridSize : f32,
    dataIntensity : f32,
    time : f32,
    canvasWidth : f32,
    canvasHeight : f32,
    
    // Extended parameters
    frequency : f32,
    phaseShift : f32,
    noiseLevel : f32,
    stripWidth : f32,
    quantumLevels : f32,
    scanSpeed : f32,
    matrixScale : f32,
    pulseRate : f32
}

@group(0) @binding(2) var<uniform> fadeParam : FadeParams;
@group(0) @binding(3) var s : sampler;
@group(0) @binding(4) var<uniform> ikeda : IkedaModeParams;

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fsFade(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    let cOld = textureSample(oldFrame, s, uv);
    let cNew = textureSample(newFrame, s, uv);
    let mixed = mix(cOld, cNew, fadeParam.fade);
    
    // Apply Ikeda processing to final mixed result
    if (ikeda.mode == 0) {
        return mixed;
    }
    
    // Smoother Black & White conversion for fade shader
    let lum = luminance(mixed.rgb);
    let dynamicThreshold = ikeda.threshold + sin(ikeda.time * 1.0) * 0.02;
    let blackWhite = step(dynamicThreshold, lum);
    
    return vec4<f32>(blackWhite, blackWhite, blackWhite, mixed.a);
}
)";

// ==================== ENHANCED PRESENT SHADER ====================

const char* ikedaPresentFragmentWGSL = R"(
@group(0) @binding(0) var oldFrame : texture_2d<f32>;
@group(0) @binding(1) var s : sampler;

struct ScrollParams {
    offset : vec2<f32>
}

struct IkedaModeParams {
    mode : i32,
    threshold : f32,
    gridSize : f32,
    dataIntensity : f32,
    time : f32,
    canvasWidth : f32,
    canvasHeight : f32,
    
    // Extended parameters
    frequency : f32,
    phaseShift : f32,
    noiseLevel : f32,
    stripWidth : f32,
    quantumLevels : f32,
    scanSpeed : f32,
    matrixScale : f32,
    pulseRate : f32
}

@group(0) @binding(2) var<uniform> scrollParam : ScrollParams;
@group(0) @binding(3) var<uniform> ikeda : IkedaModeParams;

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fsPresent(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    let uvShifted = fract(uv + scrollParam.offset);
    let baseColor = textureSample(oldFrame, s, uvShifted);
    
    if (ikeda.mode == 0) {
        return baseColor;
    }
    
    // Convert to black & white
    let lum = luminance(baseColor.rgb);
    let blackWhite = step(ikeda.threshold, lum);
    
    return vec4<f32>(blackWhite, blackWhite, blackWhite, baseColor.a);
}
)";

// ==================== DATA ANALYSIS FUNCTIONS ====================

// C++ functions to extract data from images for visualization
struct ImageAnalysisData {
    float averageLuminance;
    float variance;
    float entropy;
    std::vector<float> histogram;
    std::vector<float> edgeMap;
    uint32_t dominantFrequency;
    float compressionRatio;
};

// Function to analyze image data for Ikeda visualization
// NOTE: Commented out temporarily due to forward declaration issues
// Will be re-implemented when ImageData struct is properly forward declared
/*
ImageAnalysisData analyzeImageForIkeda(const ImageData& image) {
    ImageAnalysisData data;
    
    // Calculate basic statistics
    float sum = 0.0f;
    float sumSquared = 0.0f;
    data.histogram.resize(256, 0.0f);
    
    for (size_t i = 0; i < image.pixels.size(); i += 4) {
        uint8_t r = image.pixels[i];
        uint8_t g = image.pixels[i + 1];
        uint8_t b = image.pixels[i + 2];
        
        // Convert to luminance
        float lum = 0.299f * r + 0.587f * g + 0.114f * b;
        sum += lum;
        sumSquared += lum * lum;
        
        // Build histogram
        int bin = static_cast<int>(lum);
        if (bin >= 0 && bin < 256) {
            data.histogram[bin] += 1.0f;
        }
    }
    
    float pixelCount = static_cast<float>(image.pixels.size() / 4);
    data.averageLuminance = sum / pixelCount;
    data.variance = (sumSquared / pixelCount) - (data.averageLuminance * data.averageLuminance);
    
    // Normalize histogram
    for (auto& bin : data.histogram) {
        bin /= pixelCount;
    }
    
    // Calculate entropy
    data.entropy = 0.0f;
    for (const auto& bin : data.histogram) {
        if (bin > 0.0f) {
            data.entropy -= bin * std::log2(bin);
        }
    }
    
    // Estimate compression ratio (simplified)
    data.compressionRatio = data.entropy / 8.0f; // Rough estimate
    
    return data;
}
*/

// Export functions for JavaScript control
extern "C" {
    EMSCRIPTEN_KEEPALIVE void setIkedaMode(int mode);
    EMSCRIPTEN_KEEPALIVE void setIkedaThreshold(float threshold);
    EMSCRIPTEN_KEEPALIVE void setIkedaGridSize(float gridSize);
    EMSCRIPTEN_KEEPALIVE void setIkedaDataIntensity(float intensity);
    EMSCRIPTEN_KEEPALIVE float getImageAverageLuminance();
    EMSCRIPTEN_KEEPALIVE float getImageEntropy();
    EMSCRIPTEN_KEEPALIVE float getImageVariance();
} 