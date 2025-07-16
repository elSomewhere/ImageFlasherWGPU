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
    preprocessingMode : i32,  // 0=color, 1=black/white
    postprocessingMode : i32, // 0=grid, 1=data, 2=binary, 3=frequency, 4=scan, 5=matrix, 6=pulse, 7=noise, 8=strip, 9=phase, 10=quantum
    threshold : f32,         // black/white threshold for preprocessing
    gridSize : f32,          // grid quantization size
    dataIntensity : f32,     // data overlay intensity
    time : f32,              // global time for animations
    canvasWidth : f32,       // for precise calculations
    canvasHeight : f32,      // for precise calculations
    
    // Mode-specific parameters
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
    preprocessingMode : i32,
    postprocessingMode : i32,
    threshold : f32,
    gridSize : f32,
    dataIntensity : f32,
    time : f32,
    canvasWidth : f32,
    canvasHeight : f32,
    
    // Mode-specific parameters
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

// Restructured fragment function with preprocessing and postprocessing pipeline
@fragment
fn fsImage(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    var sampledColor = textureSample(texArr, samp, uv, u.layerIndex);
    
    // ========== PREPROCESSING STAGE ==========
    var processedColor = sampledColor;
    let lum = luminance(sampledColor.rgb);
    
    // Preprocessing Mode 0: Color with intensity and contrast controls
    if (ikeda.preprocessingMode == 0) {
        // Use threshold as contrast/brightness adjustment (-0.5 to +0.5 range)
        let contrast = ikeda.threshold * 2.0 - 1.0; // Convert 0-1 to -1 to +1 range
        let adjusted = (sampledColor.rgb - 0.5) * (1.0 + contrast) + 0.5;
        
        // Use dataIntensity as color saturation/intensity multiplier
        let intensity = ikeda.dataIntensity * 2.0; // 0-2 range for intensity
        let saturated = mix(vec3<f32>(lum), adjusted, intensity);
        
        processedColor = vec4<f32>(clamp(saturated, vec3<f32>(0.0), vec3<f32>(1.0)), sampledColor.a);
    }
    
    // Preprocessing Mode 1: Black & White conversion
    if (ikeda.preprocessingMode == 1) {
        let dynamicThreshold = ikeda.threshold + sin(ikeda.time * 1.5) * 0.05;
        let blackWhite = step(dynamicThreshold, lum);
        processedColor = vec4<f32>(blackWhite, blackWhite, blackWhite, sampledColor.a);
    }
    
    // ========== POSTPROCESSING STAGE ==========
    var finalColor = processedColor;
    
    // Postprocessing Mode 0: None (no postprocessing, just return preprocessed result)
    if (ikeda.postprocessingMode == 0) {
        finalColor = processedColor;
    }
    
    // Postprocessing Mode 1: Grid Quantization
    if (ikeda.postprocessingMode == 1) {
        let pixelSize = 1.0 / ikeda.gridSize;
        let quantizedUV = floor(uv / pixelSize) * pixelSize + pixelSize * 0.5;
        let quantizedColor = textureSample(texArr, samp, quantizedUV, u.layerIndex);
        
        if (ikeda.preprocessingMode == 1) {
            let quantLum = luminance(quantizedColor.rgb);
            let blackWhite = step(ikeda.threshold, quantLum);
            finalColor = vec4<f32>(blackWhite, blackWhite, blackWhite, sampledColor.a);
        } else {
            finalColor = quantizedColor;
        }
    }
    
    // Postprocessing Mode 2: Data Visualization Overlay
    if (ikeda.postprocessingMode == 2) {
        let baseLum = luminance(processedColor.rgb);
        
        // Simple data pattern
        let barcodeCoord = uv.x * 80.0 + baseLum * 5.0;
        let barcodePattern = step(0.5, fract(barcodeCoord + sin(ikeda.time * 0.5) * 0.1));
        
        let blendedData = barcodePattern * ikeda.dataIntensity;
        
        if (ikeda.preprocessingMode == 1) {
            let blackWhite = processedColor.r;
            let finalValue = clamp(blackWhite + blendedData * (1.0 - blackWhite), 0.0, 1.0);
            finalColor = vec4<f32>(finalValue, finalValue, finalValue, sampledColor.a);
        } else {
            finalColor = mix(processedColor, vec4<f32>(1.0, 1.0, 1.0, sampledColor.a), blendedData);
        }
    }
    
    // Postprocessing Mode 3: Binary Data Stream
    if (ikeda.postprocessingMode == 3) {
        let scanlineCount = 48.0;
        let scanlineY = floor(uv.y * scanlineCount);
        let scanlineUV = vec2<f32>(uv.x, (scanlineY + 0.5) / scanlineCount);
        let scanlineColor = textureSample(texArr, samp, scanlineUV, u.layerIndex);
        let scanLum = luminance(scanlineColor.rgb);
        
        // Binary pattern
        let binaryPattern = step(0.5, fract(uv.x * 64.0 + scanLum * 8.0));
        finalColor = vec4<f32>(binaryPattern, binaryPattern, binaryPattern, sampledColor.a);
    }
    
    // Postprocessing Mode 4: Frequency Analysis
    if (ikeda.postprocessingMode == 4) {
        let baseLum = luminance(processedColor.rgb);
        
        // Frequency bars based on luminance and frequency parameter
        let freqScale = ikeda.frequency * 2.0;
        let freqPattern = sin(uv.y * freqScale * 20.0 + ikeda.time * 2.0) * 0.5 + 0.5;
        let horizontalBars = step(0.6, freqPattern);
        
        // Spectral analysis overlay
        let spectralCoord = uv.x * freqScale * 10.0 + baseLum * 5.0;
        let spectralPattern = step(0.5, fract(spectralCoord + sin(ikeda.time) * 0.2));
        
        if (ikeda.preprocessingMode == 1) {
            let blackWhite = processedColor.r;
            let combined = clamp(blackWhite + (horizontalBars + spectralPattern) * ikeda.dataIntensity * 0.3, 0.0, 1.0);
            finalColor = vec4<f32>(combined, combined, combined, sampledColor.a);
        } else {
            let overlayIntensity = (horizontalBars + spectralPattern) * ikeda.dataIntensity * 0.3;
            finalColor = mix(processedColor, vec4<f32>(1.0, 1.0, 1.0, sampledColor.a), overlayIntensity);
        }
    }
    
    // Postprocessing Mode 5: Scan Lines
    if (ikeda.postprocessingMode == 5) {
        let baseLum = luminance(processedColor.rgb);
        
        // Progressive scanning based on scanSpeed
        let scanPosition = fract(ikeda.time * ikeda.scanSpeed);
        let scanLine = abs(uv.y - scanPosition);
        let scanEffect = step(scanLine, 0.02);
        
        // Interlaced pattern
        let interlace = step(0.5, fract(uv.y * 240.0));
        
        if (ikeda.preprocessingMode == 1) {
            let blackWhite = processedColor.r;
            let combined = clamp(blackWhite + (scanEffect + interlace * 0.3) * ikeda.dataIntensity, 0.0, 1.0);
            finalColor = vec4<f32>(combined, combined, combined, sampledColor.a);
        } else {
            let overlayIntensity = (scanEffect + interlace * 0.3) * ikeda.dataIntensity;
            finalColor = mix(processedColor, vec4<f32>(1.0, 1.0, 1.0, sampledColor.a), overlayIntensity);
        }
    }
    
    // Postprocessing Mode 6: Matrix Transformations
    if (ikeda.postprocessingMode == 6) {
        let baseLum = luminance(processedColor.rgb);
        
        // Rotating grid based on matrixScale
        let scale = ikeda.matrixScale;
        let rotation = ikeda.time * 0.5;
        let rotatedUV = vec2<f32>(
            uv.x * cos(rotation) - uv.y * sin(rotation),
            uv.x * sin(rotation) + uv.y * cos(rotation)
        );
        
        let gridPattern = step(0.9, fract(rotatedUV.x * scale * 20.0)) + 
                         step(0.9, fract(rotatedUV.y * scale * 20.0));
        
        if (ikeda.preprocessingMode == 1) {
            let blackWhite = processedColor.r;
            let combined = clamp(blackWhite + gridPattern * ikeda.dataIntensity, 0.0, 1.0);
            finalColor = vec4<f32>(combined, combined, combined, sampledColor.a);
        } else {
            finalColor = mix(processedColor, vec4<f32>(1.0, 1.0, 1.0, sampledColor.a), gridPattern * ikeda.dataIntensity);
        }
    }
    
    // Postprocessing Mode 7: Pulse Patterns
    if (ikeda.postprocessingMode == 7) {
        let baseLum = luminance(processedColor.rgb);
        
        // Radial pulse based on pulseRate
        let center = vec2<f32>(0.5, 0.5);
        let dist = distance(uv, center);
        let pulse = sin(dist * 20.0 - ikeda.time * ikeda.pulseRate * 4.0) * 0.5 + 0.5;
        let pulsePattern = step(0.7, pulse);
        
        // Rhythmic grid
        let rhythmicGrid = step(0.8, fract(uv.x * 10.0 + sin(ikeda.time * ikeda.pulseRate) * 2.0)) +
                          step(0.8, fract(uv.y * 10.0 + cos(ikeda.time * ikeda.pulseRate) * 2.0));
        
        if (ikeda.preprocessingMode == 1) {
            let blackWhite = processedColor.r;
            let combined = clamp(blackWhite + (pulsePattern + rhythmicGrid * 0.3) * ikeda.dataIntensity, 0.0, 1.0);
            finalColor = vec4<f32>(combined, combined, combined, sampledColor.a);
        } else {
            let overlayIntensity = (pulsePattern + rhythmicGrid * 0.3) * ikeda.dataIntensity;
            finalColor = mix(processedColor, vec4<f32>(1.0, 1.0, 1.0, sampledColor.a), overlayIntensity);
        }
    }
    
    // Postprocessing Mode 8: Noise Patterns
    if (ikeda.postprocessingMode == 8) {
        let baseLum = luminance(processedColor.rgb);
        
        // Structured noise based on noiseLevel
        let noiseCoord = uv * 50.0 + ikeda.time * 0.1;
        let noise1 = fract(sin(dot(noiseCoord, vec2<f32>(12.9898, 78.233))) * 43758.5453);
        let noise2 = fract(sin(dot(noiseCoord + vec2<f32>(1.0, 1.0), vec2<f32>(12.9898, 78.233))) * 43758.5453);
        
        let noisePattern = step(1.0 - ikeda.noiseLevel, noise1);
        let structuredNoise = step(0.5, noise2) * ikeda.noiseLevel;
        
        if (ikeda.preprocessingMode == 1) {
            let blackWhite = processedColor.r;
            let combined = clamp(blackWhite + (noisePattern + structuredNoise) * ikeda.dataIntensity, 0.0, 1.0);
            finalColor = vec4<f32>(combined, combined, combined, sampledColor.a);
        } else {
            let overlayIntensity = (noisePattern + structuredNoise) * ikeda.dataIntensity;
            finalColor = mix(processedColor, vec4<f32>(1.0, 1.0, 1.0, sampledColor.a), overlayIntensity);
        }
    }
    
    // Postprocessing Mode 9: Strip Decomposition
    if (ikeda.postprocessingMode == 9) {
        let baseLum = luminance(processedColor.rgb);
        
        // Alternating horizontal/vertical strips based on stripWidth
        let stripSize = ikeda.stripWidth * 10.0;
        let horizontalStrips = step(0.5, fract(uv.y / stripSize));
        let verticalStrips = step(0.5, fract(uv.x / stripSize));
        
        // Time-based switching between horizontal and vertical
        let timeSwitch = step(0.5, fract(ikeda.time * 0.3));
        let stripPattern = mix(horizontalStrips, verticalStrips, timeSwitch);
        
        if (ikeda.preprocessingMode == 1) {
            let blackWhite = processedColor.r;
            let combined = clamp(blackWhite * stripPattern + (1.0 - stripPattern) * blackWhite * 0.3, 0.0, 1.0);
            finalColor = vec4<f32>(combined, combined, combined, sampledColor.a);
        } else {
            finalColor = mix(processedColor * 0.3, processedColor, stripPattern);
        }
    }
    
    // Postprocessing Mode 10: Phase Interference
    if (ikeda.postprocessingMode == 10) {
        let baseLum = luminance(processedColor.rgb);
        
        // Multiple wave phases with interference
        let wave1 = sin(uv.x * 30.0 + ikeda.time + ikeda.phaseShift);
        let wave2 = sin(uv.y * 30.0 + ikeda.time * 1.2);
        let wave3 = sin((uv.x + uv.y) * 20.0 + ikeda.time * 0.8 + ikeda.phaseShift * 2.0);
        
        let interference = (wave1 + wave2 + wave3) / 3.0;
        let phasePattern = step(0.3, interference * 0.5 + 0.5);
        
        if (ikeda.preprocessingMode == 1) {
            let blackWhite = processedColor.r;
            let combined = clamp(blackWhite + phasePattern * ikeda.dataIntensity * 0.4, 0.0, 1.0);
            finalColor = vec4<f32>(combined, combined, combined, sampledColor.a);
        } else {
            finalColor = mix(processedColor, vec4<f32>(1.0, 1.0, 1.0, sampledColor.a), phasePattern * ikeda.dataIntensity * 0.4);
        }
    }
    
    // Postprocessing Mode 11: Quantum Levels
    if (ikeda.postprocessingMode == 11) {
        let baseLum = luminance(processedColor.rgb);
        
        // Quantize luminance to discrete levels
        let levels = ikeda.quantumLevels;
        let quantizedLum = floor(baseLum * levels) / levels;
        
        // Quantum tunneling effect
        let tunnelCoord = uv * 20.0 + ikeda.time * 0.2;
        let tunnel = fract(sin(dot(tunnelCoord, vec2<f32>(12.9898, 78.233))) * 43758.5453);
        let tunnelPattern = step(0.9, tunnel) * (1.0 / levels);
        
        // Energy level visualization
        let energyLevel = floor(quantizedLum * levels) / levels;
        let levelPattern = step(ikeda.threshold, energyLevel + tunnelPattern);
        
        finalColor = vec4<f32>(levelPattern, levelPattern, levelPattern, sampledColor.a);
    }
    
    return finalColor;
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
    preprocessingMode : i32,
    postprocessingMode : i32,
    threshold : f32,
    gridSize : f32,
    dataIntensity : f32,
    time : f32,
    canvasWidth : f32,
    canvasHeight : f32,
    
    // Mode-specific parameters
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
    
    // Apply Ikeda preprocessing to final mixed result
    if (ikeda.preprocessingMode == 0) {
        return mixed;
    }
    
    // Black & White conversion for fade shader
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
    preprocessingMode : i32,
    postprocessingMode : i32,
    threshold : f32,
    gridSize : f32,
    dataIntensity : f32,
    time : f32,
    canvasWidth : f32,
    canvasHeight : f32,
    
    // Mode-specific parameters
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
    
    if (ikeda.preprocessingMode == 0) {
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
    EMSCRIPTEN_KEEPALIVE void setPreprocessingMode(int mode);
    EMSCRIPTEN_KEEPALIVE void setPostprocessingMode(int mode);
    EMSCRIPTEN_KEEPALIVE void setIkedaThreshold(float threshold);
    EMSCRIPTEN_KEEPALIVE void setIkedaGridSize(float gridSize);
    EMSCRIPTEN_KEEPALIVE void setIkedaDataIntensity(float intensity);
    EMSCRIPTEN_KEEPALIVE float getImageAverageLuminance();
    EMSCRIPTEN_KEEPALIVE float getImageEntropy();
    EMSCRIPTEN_KEEPALIVE float getImageVariance();
} 