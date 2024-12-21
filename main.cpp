// main.cpp

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <array>
#include <chrono>
#include <cstdlib>
#include <memory>

#include <emscripten.h>
#include <emscripten/html5.h> // For emscripten_request_animation_frame_loop
#include <emscripten/html5_webgpu.h>

#include <webgpu/webgpu_cpp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

// Error handler function
void HandleUncapturedError(WGPUErrorType type, const char* message, void* userdata) {
    emscripten_log(EM_LOG_ERROR, "Uncaptured WebGPU Error (%d): %s", static_cast<int>(type), message);
}

// ==================== SHADERS ====================

// Common vertex shader: draws two triangles covering the clip-space quad
const char* vertexShaderWGSL = R"(
struct VSOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vsMain(@builtin(vertex_index) vid : u32) -> VSOutput {
    var positions = array<vec2<f32>,6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0)
    );
    var uvs = array<vec2<f32>,6>(
        vec2<f32>(0.0,1.0),
        vec2<f32>(1.0,1.0),
        vec2<f32>(1.0,0.0),
        vec2<f32>(0.0,1.0),
        vec2<f32>(1.0,0.0),
        vec2<f32>(0.0,0.0)
    );
    var out : VSOutput;
    out.Position = vec4<f32>(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}
)";

// FRAGMENT #1: Renders from ImageFlasher's 2D-array texture
const char* imageFlasherFragmentWGSL = R"(
struct Uniforms {
    layerIndex : i32
}
@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var texArr : texture_2d_array<f32>;
@group(0) @binding(2) var samp : sampler;

@fragment
fn fsImage(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(texArr, samp, uv, u.layerIndex);
}
)";

// FRAGMENT #2: Fades "oldFrame" + "newFrame" => "oldFrame"
const char* fadeFragmentWGSL = R"(
@group(0) @binding(0) var oldFrame : texture_2d<f32>;
@group(0) @binding(1) var newFrame : texture_2d<f32>;

struct FadeParams {
    fade : f32
}
@group(0) @binding(2) var<uniform> fadeParam : FadeParams;

@group(0) @binding(3) var s : sampler;

// We blend oldFrame & newFrame with factor fadeParam.fade
@fragment
fn fsFade(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    let cOld = textureSample(oldFrame, s, uv);
    let cNew = textureSample(newFrame, s, uv);
    let alpha = fadeParam.fade; // e.g. 0.5 => 80% new, 20% old
    return mix(cOld, cNew, alpha);
}
)";

// FRAGMENT #3: Just draws the "oldFrame" directly to swap chain
const char* presentFragmentWGSL = R"(
@group(0) @binding(0) var oldFrame : texture_2d<f32>;
@group(0) @binding(1) var s : sampler;

@fragment
fn fsPresent(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(oldFrame, s, uv);
}
)";

// A simple "copy" fragment: srcTex -> output
const char* copyFragmentWGSL = R"(
@group(0) @binding(0) var srcTex : texture_2d<f32>;
@group(0) @binding(1) var s : sampler;

@fragment
fn fsCopy(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(srcTex, s, uv);
}
)";

// We'll have a pipeline & bind group for that copy pass
wgpu::RenderPipeline pipelineCopy;
wgpu::BindGroup copyBindGroup;

// ========== Global WebGPU objects ==========

wgpu::Device device;
wgpu::Queue queue;
wgpu::SwapChain swapChain;
wgpu::Surface surfaceGlobal;
wgpu::TextureFormat swapChainFormat;

uint32_t g_canvasWidth = 0;
uint32_t g_canvasHeight = 0;

// We'll have three pipelines:
wgpu::RenderPipeline pipelineImageFlasher; // Renders the new image => newFrameTexture
wgpu::RenderPipeline pipelineFade;         // Fades oldFrame + newFrame => oldFrame
wgpu::RenderPipeline pipelinePresent;      // Draws oldFrame => swap chain

// We'll store fadeFactor in a uniform buffer
wgpu::Buffer fadeUniformBuffer;
wgpu::Sampler commonSampler;

// Additional texture for staging oldFrame => oldFrameTemp, so that we can read and write oldFrame in separate passes
wgpu::Texture oldFrameTempTexture;
wgpu::TextureView oldFrameTempView;

// The two “offscreen” textures for layering:
wgpu::Texture oldFrameTexture;
wgpu::TextureView oldFrameView;

wgpu::Texture newFrameTexture;
wgpu::TextureView newFrameView;

// We'll define a struct to hold "ImageData" from the decode thread, etc.
struct ImageData {
    std::vector<uint8_t> pixels;
    uint32_t width;
    uint32_t height;
};

// Thread-safe queue
template<typename T>
class ThreadSafeQueue {
public:
    void push(const T& value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(value);
        }
        condition_.notify_one();
    }

    bool tryPop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty())
            return false;
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool popBlocking(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this]{return !queue_.empty();});
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable condition_;
};

// Forward declarations
EM_BOOL frame(double time, void* userData);
void cleanup();

// We will have two queues now:
// 1. rawDataQueue: For raw PNG data from WebSocket
// 2. imageFlasher->imageQueue_: For final decoded/resized images
ThreadSafeQueue<std::vector<uint8_t>> rawDataQueue;

// decodeAndResizeImage helper function
bool decodeAndResizeImage(const uint8_t* data, int length, ImageData& imageOut) {
    int x, y, n;
    unsigned char* img = stbi_load_from_memory((const unsigned char*)data, length, &x, &y, &n, 4);
    if (!img) {
        std::cerr << "Failed to decode image from memory\n";
        return false;
    }
    const int desiredWidth = 512;
    const int desiredHeight = 512;
    std::vector<unsigned char> resized(desiredWidth * desiredHeight * 4);
    int result = stbir_resize_uint8(img, x, y, 0, resized.data(), desiredWidth, desiredHeight, 0, 4);
    stbi_image_free(img);
    if (!result) {
        std::cerr << "Failed to resize image.\n";
        return false;
    }
    imageOut.width = desiredWidth;
    imageOut.height = desiredHeight;
    imageOut.pixels = std::move(resized);
    return true;
}

// ========== ImageFlasher Class ==========
// This class organizes a ring buffer of 2D-array textures, but we only actually
// sample one image at a time. See usage in the main loop.
class ImageFlasher {
public:
    ImageFlasher(wgpu::Device device, uint32_t ringBufferSize, float imageSwitchInterval);
    ~ImageFlasher();

    void pushImage(const ImageData& image);
    void update();
    void render(wgpu::RenderPassEncoder& pass);
    void swapBuffers();
    wgpu::PipelineLayout getPipelineLayout() const { return pipelineLayout_; }

    ThreadSafeQueue<ImageData>& getImageQueue() { return imageQueue_; }

private:
    void uploadImage(const ImageData& image, int buffer);
    static const uint32_t maxLayersPerArray = 256;

    wgpu::Device device_;
    wgpu::Queue queue_;
    uint32_t ringBufferSize_;
    uint32_t textureWidth_ = 512;
    uint32_t textureHeight_ = 512;

    uint32_t writeIndex_[2];
    uint32_t displayIndex_[2];
    uint32_t imagesInBuffer_[2];

    float imageSwitchInterval_;
    std::chrono::steady_clock::time_point lastSwitchTime_[2];

    wgpu::Sampler sampler_;
    std::array<wgpu::Buffer, 2> uniformBuffers_;
    wgpu::PipelineLayout pipelineLayout_;
    wgpu::BindGroupLayout bindGroupLayout_;

    std::array<std::vector<wgpu::Texture>, 2> textureArrays_;
    std::array<std::vector<wgpu::TextureView>, 2> textureViews_;
    std::array<std::vector<wgpu::BindGroup>, 2> bindGroups_;

    ThreadSafeQueue<ImageData> imageQueue_;
    int bufferIndex_; // 0 or 1
};

ImageFlasher* imageFlasher = nullptr;

ImageFlasher::ImageFlasher(wgpu::Device device, uint32_t ringBufferSize, float imageSwitchInterval)
        : device_(device),
          queue_(device.GetQueue()),
          ringBufferSize_(ringBufferSize),
          bufferIndex_(0),
          imageSwitchInterval_(imageSwitchInterval)
{
    for (int b = 0; b < 2; ++b) {
        writeIndex_[b] = 0;
        displayIndex_[b] = 0;
        imagesInBuffer_[b] = 0;
        lastSwitchTime_[b] = std::chrono::steady_clock::now();
    }

    uint32_t numTextureArrays = (ringBufferSize_ + maxLayersPerArray - 1) / maxLayersPerArray;

    wgpu::SamplerDescriptor samplerDesc = {};
    samplerDesc.addressModeU = wgpu::AddressMode::ClampToEdge;
    samplerDesc.addressModeV = wgpu::AddressMode::ClampToEdge;
    samplerDesc.magFilter = wgpu::FilterMode::Linear;
    samplerDesc.minFilter = wgpu::FilterMode::Linear;
    sampler_ = device_.CreateSampler(&samplerDesc);

    wgpu::BindGroupLayoutEntry bglEntries[3] = {};
    bglEntries[0].binding = 0;
    bglEntries[0].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[0].buffer.type = wgpu::BufferBindingType::Uniform;
    bglEntries[0].buffer.minBindingSize = 16;

    bglEntries[1].binding = 1;
    bglEntries[1].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[1].texture.sampleType = wgpu::TextureSampleType::Float;
    bglEntries[1].texture.viewDimension = wgpu::TextureViewDimension::e2DArray;
    bglEntries[1].texture.multisampled = false;

    bglEntries[2].binding = 2;
    bglEntries[2].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[2].sampler.type = wgpu::SamplerBindingType::Filtering;

    wgpu::BindGroupLayoutDescriptor bglDesc = {};
    bglDesc.entryCount = 3;
    bglDesc.entries = bglEntries;

    bindGroupLayout_ = device_.CreateBindGroupLayout(&bglDesc);

    wgpu::PipelineLayoutDescriptor plDesc = {};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &bindGroupLayout_;
    pipelineLayout_ = device_.CreatePipelineLayout(&plDesc);

    for (int b = 0; b < 2; ++b) {
        wgpu::BufferDescriptor uniformBufferDesc = {};
        uniformBufferDesc.size = 16;
        uniformBufferDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        uniformBuffers_[b] = device_.CreateBuffer(&uniformBufferDesc);

        textureArrays_[b].resize(numTextureArrays);
        textureViews_[b].resize(numTextureArrays);
        bindGroups_[b].resize(numTextureArrays);

        for (uint32_t i = 0; i < numTextureArrays; ++i) {
            uint32_t layersInThisArray = maxLayersPerArray;
            if (i == numTextureArrays - 1 && (ringBufferSize_ % maxLayersPerArray != 0)) {
                layersInThisArray = ringBufferSize_ % maxLayersPerArray;
            }

            wgpu::TextureDescriptor textureDesc = {};
            textureDesc.size.width = textureWidth_;
            textureDesc.size.height = textureHeight_;
            textureDesc.size.depthOrArrayLayers = layersInThisArray;
            textureDesc.format = wgpu::TextureFormat::RGBA8Unorm;
            textureDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
            wgpu::Texture textureArray = device_.CreateTexture(&textureDesc);
            textureArrays_[b][i] = textureArray;

            wgpu::TextureViewDescriptor viewDesc = {};
            viewDesc.dimension = wgpu::TextureViewDimension::e2DArray;
            textureViews_[b][i] = textureArray.CreateView(&viewDesc);

            wgpu::BindGroupEntry bgEntries[3] = {};
            bgEntries[0].binding = 0;
            bgEntries[0].buffer = uniformBuffers_[b];
            bgEntries[0].offset = 0;
            bgEntries[0].size = 16;

            bgEntries[1].binding = 1;
            bgEntries[1].textureView = textureViews_[b][i];

            bgEntries[2].binding = 2;
            bgEntries[2].sampler = sampler_;

            wgpu::BindGroupDescriptor bgDesc = {};
            bgDesc.layout = bindGroupLayout_;
            bgDesc.entryCount = 3;
            bgDesc.entries = bgEntries;

            bindGroups_[b][i] = device_.CreateBindGroup(&bgDesc);
        }
    }
}

ImageFlasher::~ImageFlasher() {
    std::cout << "ImageFlasher destroyed." << std::endl;
}

void ImageFlasher::pushImage(const ImageData& image) {
    imageQueue_.push(image);
}

void ImageFlasher::uploadImage(const ImageData& image, int buffer) {
    if (imagesInBuffer_[buffer] == ringBufferSize_) {
        // Overwrite oldest
        displayIndex_[buffer] = (displayIndex_[buffer] + 1) % ringBufferSize_;
    } else {
        imagesInBuffer_[buffer]++;
    }

    uint32_t globalLayerIndex = writeIndex_[buffer];
    uint32_t textureArrayIndex = globalLayerIndex / maxLayersPerArray;
    uint32_t layerIndexInTexture = globalLayerIndex % maxLayersPerArray;

    wgpu::ImageCopyTexture dst = {};
    dst.texture = textureArrays_[buffer][textureArrayIndex];
    dst.mipLevel = 0;
    dst.origin = {0, 0, layerIndexInTexture};
    dst.aspect = wgpu::TextureAspect::All;

    wgpu::TextureDataLayout layout = {};
    layout.offset = 0;
    layout.bytesPerRow = image.width * 4;
    layout.rowsPerImage = image.height;

    wgpu::Extent3D extent = {};
    extent.width = image.width;
    extent.height = image.height;
    extent.depthOrArrayLayers = 1;

    queue_.WriteTexture(&dst, image.pixels.data(), image.pixels.size(), &layout, &extent);

    writeIndex_[buffer] = (writeIndex_[buffer] + 1) % ringBufferSize_;
}

void ImageFlasher::update() {
    int frontBuffer = bufferIndex_;
    int backBuffer = 1 - frontBuffer;

    bool uploadedAnyImage = false;
    while (true) {
        ImageData image;
        if (!imageQueue_.tryPop(image)) {
            break;
        }
        uploadImage(image, backBuffer);
        uploadedAnyImage = true;
    }

    if (uploadedAnyImage) {
        swapBuffers();
        frontBuffer = bufferIndex_;
    }

    if (imagesInBuffer_[frontBuffer] > 0) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed = now - lastSwitchTime_[frontBuffer];
        if (elapsed.count() >= imageSwitchInterval_) {
            displayIndex_[frontBuffer] = (displayIndex_[frontBuffer] + 1) % imagesInBuffer_[frontBuffer];
            lastSwitchTime_[frontBuffer] = now;

            uint32_t globalLayerIndex = displayIndex_[frontBuffer];
            uint32_t layerIndexInTexture = globalLayerIndex % maxLayersPerArray;

            struct UniformsData {
                int32_t layerIndex;
                int32_t padding[3];
            };
            UniformsData uniformsData = { (int32_t)layerIndexInTexture, {0,0,0} };
            queue_.WriteBuffer(uniformBuffers_[frontBuffer], 0, &uniformsData, sizeof(UniformsData));
        }
    }
}

void ImageFlasher::render(wgpu::RenderPassEncoder& pass) {
    int frontBuffer = bufferIndex_;
    if (imagesInBuffer_[frontBuffer] == 0) {
        // No images loaded yet; just bind the 0th
        pass.SetBindGroup(0, bindGroups_[frontBuffer][0]);
        return;
    }

    uint32_t globalLayerIndex = displayIndex_[frontBuffer] % ringBufferSize_;
    uint32_t textureArrayIndex = globalLayerIndex / maxLayersPerArray;
    pass.SetBindGroup(0, bindGroups_[frontBuffer][textureArrayIndex]);
}

void ImageFlasher::swapBuffers() {
    bufferIndex_ = 1 - bufferIndex_;
    lastSwitchTime_[bufferIndex_] = std::chrono::steady_clock::now();

    if (imagesInBuffer_[bufferIndex_] > 0) {
        uint32_t globalLayerIndex = displayIndex_[bufferIndex_];
        uint32_t layerIndexInTexture = globalLayerIndex % maxLayersPerArray;

        struct UniformsData {
            int32_t layerIndex;
            int32_t padding[3];
        };
        UniformsData uniformsData = { (int32_t)layerIndexInTexture, {0,0,0} };
        queue_.WriteBuffer(uniformBuffers_[bufferIndex_], 0, &uniformsData, sizeof(UniformsData));
    }
}

// ========== Helpers to create Shaders/Textures/etc. ==========

wgpu::ShaderModule createShaderModule(const char* code) {
    wgpu::ShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.code = code;

    wgpu::ShaderModuleDescriptor desc = {};
    desc.nextInChain = &wgslDesc;

    return device.CreateShaderModule(&desc);
}

void createOffscreenTextures(uint32_t w, uint32_t h) {
    // oldFrameTexture
    {
        wgpu::TextureDescriptor desc = {};
        desc.size.width = w;
        desc.size.height = h;
        desc.size.depthOrArrayLayers = 1;
        desc.format = wgpu::TextureFormat::RGBA8Unorm;
        desc.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
        oldFrameTexture = device.CreateTexture(&desc);
        oldFrameView = oldFrameTexture.CreateView();
    }

    // newFrameTexture
    {
        wgpu::TextureDescriptor desc = {};
        desc.size.width = w;
        desc.size.height = h;
        desc.size.depthOrArrayLayers = 1;
        desc.format = wgpu::TextureFormat::RGBA8Unorm;
        desc.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
        newFrameTexture = device.CreateTexture(&desc);
        newFrameView = newFrameTexture.CreateView();
    }

    // oldFrameTempTexture
    {
        wgpu::TextureDescriptor desc = {};
        desc.size.width = w;
        desc.size.height = h;
        desc.size.depthOrArrayLayers = 1;
        desc.format = wgpu::TextureFormat::RGBA8Unorm;
        desc.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
        oldFrameTempTexture = device.CreateTexture(&desc);
        oldFrameTempView = oldFrameTempTexture.CreateView();
    }
}

// ========== Create Pipelines ==========

void createPipelineCopy() {
    wgpu::ShaderModule vs = createShaderModule(vertexShaderWGSL);
    wgpu::ShaderModule fs = createShaderModule(copyFragmentWGSL);

    // Bind group layout for pipelineCopy
    wgpu::BindGroupLayoutEntry bglEntries[2] = {};
    bglEntries[0].binding = 0;
    bglEntries[0].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[0].texture.sampleType = wgpu::TextureSampleType::Float;
    bglEntries[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;

    bglEntries[1].binding = 1;
    bglEntries[1].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[1].sampler.type = wgpu::SamplerBindingType::Filtering;

    wgpu::BindGroupLayoutDescriptor bglDesc = {};
    bglDesc.entryCount = 2;
    bglDesc.entries = bglEntries;
    wgpu::BindGroupLayout copyBGL = device.CreateBindGroupLayout(&bglDesc);

    wgpu::PipelineLayoutDescriptor pld = {};
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts = &copyBGL;
    wgpu::PipelineLayout layout = device.CreatePipelineLayout(&pld);

    wgpu::RenderPipelineDescriptor desc = {};
    desc.layout = layout;
    desc.vertex.module = vs;
    desc.vertex.entryPoint = "vsMain";

    wgpu::ColorTargetState ct = {};
    ct.format = wgpu::TextureFormat::RGBA8Unorm;
    ct.writeMask = wgpu::ColorWriteMask::All;

    wgpu::FragmentState fsState = {};
    fsState.module = fs;
    fsState.entryPoint = "fsCopy";
    fsState.targetCount = 1;
    fsState.targets = &ct;
    desc.fragment = &fsState;

    desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    desc.primitive.cullMode = wgpu::CullMode::None;
    desc.multisample.count = 1;

    pipelineCopy = device.CreateRenderPipeline(&desc);
}

void createPipelineImageFlasher() {
    // Renders from the ImageFlasher's 2D array => "newFrameTexture"
    wgpu::ShaderModule vs = createShaderModule(vertexShaderWGSL);
    wgpu::ShaderModule fs = createShaderModule(imageFlasherFragmentWGSL);

    // We'll use the layout from the imageFlasher
    wgpu::PipelineLayout layout = imageFlasher->getPipelineLayout();

    wgpu::RenderPipelineDescriptor desc = {};
    desc.layout = layout;
    desc.vertex.module = vs;
    desc.vertex.entryPoint = "vsMain";

    wgpu::ColorTargetState colorTarget = {};
    colorTarget.format = wgpu::TextureFormat::RGBA8Unorm;
    colorTarget.writeMask = wgpu::ColorWriteMask::All;

    wgpu::FragmentState fsState = {};
    fsState.module = fs;
    fsState.entryPoint = "fsImage";
    fsState.targetCount = 1;
    fsState.targets = &colorTarget;

    desc.fragment = &fsState;
    desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    desc.primitive.cullMode = wgpu::CullMode::None;
    desc.multisample.count = 1;

    pipelineImageFlasher = device.CreateRenderPipeline(&desc);
}

void createPipelineFade() {
    // oldFrame + newFrame => oldFrame
    wgpu::ShaderModule vs = createShaderModule(vertexShaderWGSL);
    wgpu::ShaderModule fs = createShaderModule(fadeFragmentWGSL);

    wgpu::BindGroupLayoutEntry bglEntries[4] = {};
    // oldFrame
    bglEntries[0].binding = 0;
    bglEntries[0].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[0].texture.sampleType = wgpu::TextureSampleType::Float;
    bglEntries[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;

    // newFrame
    bglEntries[1].binding = 1;
    bglEntries[1].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[1].texture.sampleType = wgpu::TextureSampleType::Float;
    bglEntries[1].texture.viewDimension = wgpu::TextureViewDimension::e2D;

    // fadeParam
    bglEntries[2].binding = 2;
    bglEntries[2].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[2].buffer.type = wgpu::BufferBindingType::Uniform;

    // sampler
    bglEntries[3].binding = 3;
    bglEntries[3].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[3].sampler.type = wgpu::SamplerBindingType::Filtering;

    wgpu::BindGroupLayoutDescriptor bglDesc = {};
    bglDesc.entryCount = 4;
    bglDesc.entries = bglEntries;
    wgpu::BindGroupLayout fadeBGL = device.CreateBindGroupLayout(&bglDesc);

    wgpu::PipelineLayoutDescriptor plDesc = {};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &fadeBGL;
    wgpu::PipelineLayout layout = device.CreatePipelineLayout(&plDesc);

    wgpu::RenderPipelineDescriptor desc = {};
    desc.layout = layout;
    desc.vertex.module = vs;
    desc.vertex.entryPoint = "vsMain";

    wgpu::ColorTargetState colorTarget = {};
    colorTarget.format = wgpu::TextureFormat::RGBA8Unorm; // We render into oldFrameTexture
    colorTarget.writeMask = wgpu::ColorWriteMask::All;

    wgpu::FragmentState fsState = {};
    fsState.module = fs;
    fsState.entryPoint = "fsFade";
    fsState.targetCount = 1;
    fsState.targets = &colorTarget;

    desc.fragment = &fsState;
    desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    desc.primitive.cullMode = wgpu::CullMode::None;
    desc.multisample.count = 1;

    pipelineFade = device.CreateRenderPipeline(&desc);

    // Create fadeUniformBuffer
    {
        wgpu::BufferDescriptor bd = {};
        bd.size = sizeof(float);
        bd.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        fadeUniformBuffer = device.CreateBuffer(&bd);

        float fadeFactor = 0.5f; // default fade factor
        queue.WriteBuffer(fadeUniformBuffer, 0, &fadeFactor, sizeof(fadeFactor));
    }
}

void createPipelinePresent() {
    // oldFrame => swap chain
    wgpu::ShaderModule vs = createShaderModule(vertexShaderWGSL);
    wgpu::ShaderModule fs = createShaderModule(presentFragmentWGSL);

    wgpu::BindGroupLayoutEntry bglEntries[2] = {};
    // oldFrame
    bglEntries[0].binding = 0;
    bglEntries[0].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[0].texture.sampleType = wgpu::TextureSampleType::Float;
    bglEntries[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;

    // sampler
    bglEntries[1].binding = 1;
    bglEntries[1].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[1].sampler.type = wgpu::SamplerBindingType::Filtering;

    wgpu::BindGroupLayoutDescriptor bglDesc = {};
    bglDesc.entryCount = 2;
    bglDesc.entries = bglEntries;
    wgpu::BindGroupLayout presentBGL = device.CreateBindGroupLayout(&bglDesc);

    wgpu::PipelineLayoutDescriptor plDesc = {};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &presentBGL;
    wgpu::PipelineLayout layout = device.CreatePipelineLayout(&plDesc);

    wgpu::RenderPipelineDescriptor desc = {};
    desc.layout = layout;
    desc.vertex.module = vs;
    desc.vertex.entryPoint = "vsMain";

    wgpu::ColorTargetState ct = {};
    ct.format = swapChainFormat; // final output
    ct.writeMask = wgpu::ColorWriteMask::All;

    wgpu::FragmentState fsState = {};
    fsState.module = fs;
    fsState.entryPoint = "fsPresent";
    fsState.targetCount = 1;
    fsState.targets = &ct;

    desc.fragment = &fsState;
    desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    desc.primitive.cullMode = wgpu::CullMode::None;
    desc.multisample.count = 1;

    pipelinePresent = device.CreateRenderPipeline(&desc);
}

// We'll also create the bind groups for fade + present in the frame loop (or on init).
// For convenience, we’ll create them dynamically each frame referencing the current views
// or create re-usable ones if the textures aren't changing.
// In the code below, we create them inline in each pass, but you can also cache them.

extern "C" void initializeSwapChainAndPipeline(wgpu::Surface surface) {
    wgpu::SwapChainDescriptor scDesc = {};
    scDesc.format = wgpu::TextureFormat::BGRA8Unorm;
    scDesc.usage = wgpu::TextureUsage::RenderAttachment;
    scDesc.presentMode = wgpu::PresentMode::Fifo;

    double cw, ch;
    emscripten_get_element_css_size("canvas", &cw, &ch);
    g_canvasWidth = (uint32_t)cw;
    g_canvasHeight = (uint32_t)ch;

    scDesc.width = g_canvasWidth;
    scDesc.height = g_canvasHeight;

    swapChain = device.CreateSwapChain(surface, &scDesc);
    if (!swapChain) {
        std::cerr << "Failed to create swap chain.\n";
        return;
    }
    swapChainFormat = scDesc.format;

    // Create ImageFlasher
    constexpr uint32_t RING_BUFFER_SIZE = 1024;
    constexpr float IMAGE_SWITCH_INTERVAL = 1.0f/3.0f;
    imageFlasher = new ImageFlasher(device, RING_BUFFER_SIZE, IMAGE_SWITCH_INTERVAL);

    // Create pipelines
    createPipelineImageFlasher();
    createPipelineFade();
    createPipelinePresent();
    createPipelineCopy();

    // Create offscreen textures
    createOffscreenTextures(g_canvasWidth, g_canvasHeight);

    // Common sampler if not already
    {
        wgpu::SamplerDescriptor sd = {};
        sd.minFilter = wgpu::FilterMode::Linear;
        sd.magFilter = wgpu::FilterMode::Linear;
        sd.addressModeU = wgpu::AddressMode::ClampToEdge;
        sd.addressModeV = wgpu::AddressMode::ClampToEdge;
        commonSampler = device.CreateSampler(&sd);
    }

    // Start the main loop via request_animation_frame_loop
    emscripten_request_animation_frame_loop([](double /*time*/, void*) {
        // 1) Acquire swapChainView
        wgpu::TextureView swapChainView = swapChain.GetCurrentTextureView();
        if(!swapChainView) return EM_TRUE;

        // 2) Update flasher
        imageFlasher->update();

        wgpu::CommandEncoder encoder = device.CreateCommandEncoder({});

        // ========== Pass #1: "newFrame pass" ==========
        {
            wgpu::RenderPassColorAttachment att = {};
            att.view = newFrameView;
            att.loadOp = wgpu::LoadOp::Clear;
            att.storeOp = wgpu::StoreOp::Store;
            att.clearValue = {0,0,0,1};

            wgpu::RenderPassDescriptor desc = {};
            desc.colorAttachmentCount = 1;
            desc.colorAttachments = &att;

            wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&desc);
            pass.SetPipeline(pipelineImageFlasher);
            imageFlasher->render(pass);
            pass.Draw(6);
            pass.End();
        }

        // ========== Pass #2: copy oldFrame => oldFrameTemp ==========
        {
            // We'll build a short-lived bind group for the copy pipeline:
            wgpu::BindGroup copyBG = [&]{
                wgpu::BindGroupLayout bgl = pipelineCopy.GetBindGroupLayout(0);

                wgpu::BindGroupEntry entries[2] = {};
                entries[0].binding = 0;  // srcTex
                entries[0].textureView = oldFrameView;
                entries[1].binding = 1;  // sampler
                entries[1].sampler = commonSampler;

                wgpu::BindGroupDescriptor bd = {};
                bd.layout = bgl;
                bd.entryCount = 2;
                bd.entries = entries;
                return device.CreateBindGroup(&bd);
            }();

            wgpu::RenderPassColorAttachment att = {};
            att.view = oldFrameTempView;
            att.loadOp = wgpu::LoadOp::Clear;
            att.storeOp = wgpu::StoreOp::Store;
            att.clearValue = {0,0,0,1};

            wgpu::RenderPassDescriptor desc = {};
            desc.colorAttachmentCount = 1;
            desc.colorAttachments = &att;

            wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&desc);
            pass.SetPipeline(pipelineCopy);
            pass.SetBindGroup(0, copyBG);
            pass.Draw(6);
            pass.End();
        }

        // ========== Pass #3: fade pass => oldFrame ( blending oldFrameTemp & newFrame ) ==========
        {
            // Another short-lived bind group for the fade pipeline
            wgpu::BindGroup fadeBG = [&]{
                wgpu::BindGroupLayout bgl = pipelineFade.GetBindGroupLayout(0);
                wgpu::BindGroupEntry e[4] = {};
                // oldFrame => oldFrameTempView
                e[0].binding = 0;
                e[0].textureView = oldFrameTempView;
                // newFrame
                e[1].binding = 1;
                e[1].textureView = newFrameView;
                // fadeParam
                e[2].binding = 2;
                e[2].buffer = fadeUniformBuffer;
                e[2].size = sizeof(float);
                // sampler
                e[3].binding = 3;
                e[3].sampler = commonSampler;

                wgpu::BindGroupDescriptor bd = {};
                bd.layout = bgl;
                bd.entryCount = 4;
                bd.entries = e;
                return device.CreateBindGroup(&bd);
            }();

            wgpu::RenderPassColorAttachment att = {};
            att.view = oldFrameView;     // Overwrite oldFrame with fade result
            att.loadOp = wgpu::LoadOp::Clear;
            att.storeOp = wgpu::StoreOp::Store;
            att.clearValue = {0,0,0,1};

            wgpu::RenderPassDescriptor desc = {};
            desc.colorAttachmentCount = 1;
            desc.colorAttachments = &att;

            wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&desc);
            pass.SetPipeline(pipelineFade);
            pass.SetBindGroup(0, fadeBG);
            pass.Draw(6);
            pass.End();
        }

        // ========== Pass #4: present pass => oldFrame => swapChain ==========
        {
            wgpu::RenderPassColorAttachment att = {};
            att.view = swapChainView;
            att.loadOp = wgpu::LoadOp::Clear;
            att.storeOp = wgpu::StoreOp::Store;
            att.clearValue = {0.3f, 0.3f, 0.3f, 1.0f};

            wgpu::RenderPassDescriptor desc = {};
            desc.colorAttachmentCount = 1;
            desc.colorAttachments = &att;

            // Another short-lived bind group for the present pipeline
            wgpu::BindGroup presentBG = [&] {
                wgpu::BindGroupLayout bgl = pipelinePresent.GetBindGroupLayout(0);
                wgpu::BindGroupEntry e[2] = {};
                e[0].binding = 0;
                e[0].textureView = oldFrameView;  // reading from oldFrame
                e[1].binding = 1;
                e[1].sampler = commonSampler;

                wgpu::BindGroupDescriptor bd = {};
                bd.layout = bgl;
                bd.entryCount = 2;
                bd.entries = e;
                return device.CreateBindGroup(&bd);
            }();

            wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&desc);
            pass.SetPipeline(pipelinePresent);
            pass.SetBindGroup(0, presentBG);
            pass.Draw(6);
            pass.End();
        }

        wgpu::CommandBuffer cmd = encoder.Finish();
        queue.Submit(1, &cmd);

        return EM_TRUE;
    }, nullptr);
}

// ========== Worker thread to decode & resize images ==========

std::atomic<bool> decodeWorkerRunning(true);
std::thread decodeWorkerThread;

void decodeWorkerFunc() {
    while (decodeWorkerRunning) {
        std::vector<uint8_t> rawData;
        if (!rawDataQueue.popBlocking(rawData)) {
            continue; // Shouldn't happen due to blocking pop
        }

        ImageData imgData;
        if (!decodeAndResizeImage(rawData.data(), (int)rawData.size(), imgData)) {
            continue; // Failed to decode/resize
        }

        if (imageFlasher) {
            imageFlasher->pushImage(imgData);
        }
    }
}

// Called from JS when we receive image data over WebSocket
extern "C" {
EMSCRIPTEN_KEEPALIVE
void onImageReceived(uint8_t* data, int length) {
    // Instead of decoding here, just push raw data to rawDataQueue
    std::vector<uint8_t> raw(data, data + length);
    rawDataQueue.push(raw);
}
}

void cleanup() {
    decodeWorkerRunning = false;
    if (decodeWorkerThread.joinable()) {
        decodeWorkerThread.join();
    }
    delete imageFlasher;
    imageFlasher = nullptr;
}

// ========== Device/Adapter callbacks ==========

void onDeviceRequestEnded(WGPURequestDeviceStatus status, WGPUDevice cDevice, const char* message, void* userdata) {
    if (status == WGPURequestDeviceStatus_Success) {
        device = wgpu::Device::Acquire(cDevice);
        queue = device.GetQueue();
        device.SetUncapturedErrorCallback(HandleUncapturedError, nullptr);

        WGPUSurface surface = (WGPUSurface)userdata;
        initializeSwapChainAndPipeline(wgpu::Surface::Acquire(surface));

        // Start decode worker now that imageFlasher is ready
        decodeWorkerThread = std::thread(decodeWorkerFunc);

    } else {
        std::cerr << "Failed to create device: " << (message ? message : "Unknown error") << std::endl;
    }
}

void onAdapterRequestEnded(WGPURequestAdapterStatus status, WGPUAdapter cAdapter, const char* message, void* userdata) {
    if (status == WGPURequestAdapterStatus_Success) {
        wgpu::Adapter adapter = wgpu::Adapter::Acquire(cAdapter);
        wgpu::DeviceDescriptor deviceDesc = {};
        deviceDesc.label = "My Device";
        adapter.RequestDevice(&deviceDesc, onDeviceRequestEnded, userdata);
    } else {
        std::cerr << "Failed to get WebGPU adapter: " << (message ? message : "Unknown error") << std::endl;
    }
}

// ========== main() ==========

int main() {
    WGPUInstanceDescriptor instanceDesc = {};
    WGPUInstance instance = wgpuCreateInstance(&instanceDesc);

    WGPURequestAdapterOptions opts = {};
    opts.powerPreference = WGPUPowerPreference_HighPerformance;

    WGPUSurfaceDescriptorFromCanvasHTMLSelector canv = {};
    canv.chain.sType = WGPUSType_SurfaceDescriptorFromCanvasHTMLSelector;
    canv.selector = "canvas";

    WGPUSurfaceDescriptor surfDesc = {};
    surfDesc.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&canv);

    WGPUSurface surface = wgpuInstanceCreateSurface(instance, &surfDesc);
    if (!surface) {
        std::cerr << "Failed to create surface.\n";
        return -1;
    }
    surfaceGlobal = wgpu::Surface::Acquire(surface);

    wgpuInstanceRequestAdapter(instance, &opts, onAdapterRequestEnded, surface);

    emscripten_exit_with_live_runtime();
    return 0;
}
