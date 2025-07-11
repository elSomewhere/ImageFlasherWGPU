// Enhanced Ikeda-Inspired Shaders for ImageFlasherWGPU
// Phase 1: Black & White conversion, Grid quantization, Data visualization
// 
// Bug Fixes Applied:
// - Grid Mode: Fixed inverted grid line density (larger grid size = fewer lines)
// - Data Mode: Replaced harsh max() with smooth blending to prevent sudden transitions
// - All Modes: Reduced flickering and improved visual stability
// - Binary Mode: Better scanline resolution and bit extraction stability

// ==================== IKEDA MODE UNIFORMS ====================

const char* ikedaModeUniformWGSL = R"(
struct IkedaModeParams {
    mode : i32,              // 0=normal, 1=blackwhite, 2=grid, 3=data, 4=binary
    threshold : f32,         // black/white threshold
    gridSize : f32,          // grid quantization size
    dataIntensity : f32,     // data overlay intensity
    time : f32,              // global time for animations
    canvasWidth : f32,       // for precise calculations
    canvasHeight : f32,      // for precise calculations
    padding : f32            // alignment
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
    padding : f32
}

@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var texArr : texture_2d_array<f32>;
@group(0) @binding(2) var samp : sampler;
@group(0) @binding(3) var<uniform> ikeda : IkedaModeParams;

// Precise luminance calculation
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

// Grid quantization function
fn quantizeToGrid(uv: vec2<f32>, gridSize: f32) -> vec2<f32> {
    let pixelSize = 1.0 / gridSize;
    return floor(uv / pixelSize) * pixelSize + pixelSize * 0.5;
}

// Generate smooth data pattern based on pixel values
fn generateDataPattern(uv: vec2<f32>, color: vec3<f32>, time: f32) -> f32 {
    let lum = luminance(color);
    
    // Smooth barcode-like pattern
    let barcodeCoord = uv.x * 80.0 + lum * 5.0;
    let barcodePattern = smoothstep(0.4, 0.6, fract(barcodeCoord + sin(time * 0.5) * 0.1));
    
    // Smooth grid overlay (fixed size for data mode)
    let gridCoord = uv * 48.0;
    let gridThickness = 0.08;
    let gridX = smoothstep(1.0 - gridThickness, 1.0, fract(gridCoord.x));
    let gridY = smoothstep(1.0 - gridThickness, 1.0, fract(gridCoord.y));
    let gridPattern = max(gridX, gridY);
    
    // Smooth binary data visualization
    let binaryCoord = floor(uv * 24.0);
    let binaryValue = fract(sin(dot(binaryCoord, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let binaryPattern = smoothstep(lum - 0.1, lum + 0.1, binaryValue);
    
    // Smooth combination instead of harsh max()
    let combinedPattern = barcodePattern * 0.4 + gridPattern * 0.3 + binaryPattern * 0.3;
    return clamp(combinedPattern, 0.0, 1.0);
}

// Main fragment function with Ikeda modes
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
        // Reduced dynamic threshold variation to prevent excessive flickering
        let dynamicThreshold = ikeda.threshold + sin(ikeda.time * 1.5) * 0.05;
        let blackWhite = step(dynamicThreshold, lum);
        return vec4<f32>(blackWhite, blackWhite, blackWhite, sampledColor.a);
    }
    
    // Mode 2: Grid Quantization
    if (ikeda.mode == 2) {
        let quantizedUV = quantizeToGrid(uv, ikeda.gridSize);
        let quantizedColor = textureSample(texArr, samp, quantizedUV, u.layerIndex);
        let lum = luminance(quantizedColor.rgb);
        let blackWhite = step(ikeda.threshold, lum);
        
        // Return only the quantized result without grid lines
        return vec4<f32>(blackWhite, blackWhite, blackWhite, sampledColor.a);
    }
    
    // Mode 3: Data Visualization Overlay
    if (ikeda.mode == 3) {
        let lum = luminance(sampledColor.rgb);
        let blackWhite = step(ikeda.threshold, lum);
        
        let dataPattern = generateDataPattern(uv, sampledColor.rgb, ikeda.time);
        
        // Smooth blending instead of harsh max() - prevents sudden transitions
        let blendedData = dataPattern * ikeda.dataIntensity;
        let finalValue = clamp(blackWhite + blendedData * (1.0 - blackWhite), 0.0, 1.0);
        
        return vec4<f32>(finalValue, finalValue, finalValue, sampledColor.a);
    }
    
    // Mode 4: Binary Data Stream
    if (ikeda.mode == 4) {
        // Convert to scanlines with better resolution
        let scanlineCount = 48.0; // Balanced resolution
        let scanlineY = floor(uv.y * scanlineCount);
        let scanlineUV = vec2<f32>(uv.x, (scanlineY + 0.5) / scanlineCount);
        let scanlineColor = textureSample(texArr, samp, scanlineUV, u.layerIndex);
        
        // Create more stable binary representation
        let lum = luminance(scanlineColor.rgb);
        let pixelValue = u32(lum * 255.0);
        let bitsPerPixel = 8.0;
        let bitPosition = u32(floor(fract(uv.x * bitsPerPixel) * bitsPerPixel));
        let bitValue = f32((pixelValue >> bitPosition) & 1u);
        
        return vec4<f32>(bitValue, bitValue, bitValue, sampledColor.a);
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
    padding : f32
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
    let dynamicThreshold = ikeda.threshold + sin(ikeda.time * 1.0) * 0.02; // Less aggressive
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
    padding : f32
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
    
    // No edge overlay patterns - just return the black & white conversion
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