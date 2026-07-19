/*********************
main.cpp
*********************/

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
#include <cmath>       // for fmod, floor, etc.
#include <algorithm>   // for std::swap
#include <unordered_set>
// #include <random>   // Removed std::shuffle usage to avoid the compile error

#include <emscripten.h>
#include <emscripten/html5.h> // For emscripten_request_animation_frame_loop

#include <webgpu/webgpu_cpp.h>

// stb_image for decode
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// stb_image_resize for resize
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

// Include Ikeda shaders
#include "ikeda_shaders.cpp"

// ==================== SHADERS ====================

// A full-screen quad vertex shader
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

const char* tileVertexShaderWGSL = R"(
struct TileBuffer {
    gridSize : u32,
    residentCount : u32,
    pad0 : u32,
    pad1 : u32,
    layers : array<u32>,
}

@group(0) @binding(0) var<storage, read> tiles : TileBuffer;

struct VSOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) uv : vec2<f32>,
    @location(1) @interpolate(flat) layerIndex : i32,
    @location(2) @interpolate(flat) instanceId : u32,
};

@vertex
fn vsTile(@builtin(vertex_index) vid : u32, @builtin(instance_index) instance : u32) -> VSOutput {
    var positions = array<vec2<f32>,6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
    );
    var uvs = array<vec2<f32>,6>(
        vec2<f32>(0.0,1.0), vec2<f32>(1.0,1.0), vec2<f32>(1.0,0.0),
        vec2<f32>(0.0,1.0), vec2<f32>(1.0,0.0), vec2<f32>(0.0,0.0)
    );
    let grid = max(tiles.gridSize, 1u);
    let column = instance % grid;
    let row = instance / grid;
    let tileScale = 1.0 / f32(grid);
    let center = vec2<f32>(
        -1.0 + (f32(column) * 2.0 + 1.0) * tileScale,
         1.0 - (f32(row) * 2.0 + 1.0) * tileScale
    );
    var out : VSOutput;
    out.Position = vec4<f32>(center + positions[vid] * tileScale, 0.0, 1.0);
    out.uv = uvs[vid];
    out.layerIndex = i32(tiles.layers[instance]);
    out.instanceId = instance;
    return out;
}
)";

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

const char* fadeFragmentWGSL = R"(
@group(0) @binding(0) var oldFrame : texture_2d<f32>;
@group(0) @binding(1) var newFrame : texture_2d<f32>;

struct FadeParams {
    fade : f32
}
@group(0) @binding(2) var<uniform> fadeParam : FadeParams;

@group(0) @binding(3) var s : sampler;

@fragment
fn fsFade(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    let cOld = textureSample(oldFrame, s, uv);
    let cNew = textureSample(newFrame, s, uv);
    let alpha = fadeParam.fade;
    return mix(cOld, cNew, alpha);
}
)";

const char* presentFragmentWGSL = R"(
@group(0) @binding(0) var oldFrame : texture_2d<f32>;
@group(0) @binding(1) var s : sampler;

struct ScrollParams {
    offset : vec2<f32>
}

@group(0) @binding(2) var<uniform> scrollParam : ScrollParams;

@fragment
fn fsPresent(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    let uvShifted = fract(uv + scrollParam.offset);
    return textureSample(oldFrame, s, uvShifted);
}
)";

const char* copyFragmentWGSL = R"(
@group(0) @binding(0) var srcTex : texture_2d<f32>;
@group(0) @binding(1) var s : sampler;

@fragment
fn fsCopy(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(srcTex, s, uv);
}
)";

// ========== Global WebGPU objects & Buffers ==========

wgpu::Device device;
wgpu::Queue queue;
wgpu::Instance instance = wgpuCreateInstance(nullptr);
wgpu::Adapter adapter;
wgpu::Surface surfaceGlobal;
wgpu::TextureFormat swapChainFormat;

uint32_t g_canvasWidth = 0;
uint32_t g_canvasHeight = 0;

wgpu::RenderPipeline pipelineImageFlasher;
wgpu::RenderPipeline pipelineFade;
wgpu::RenderPipeline pipelinePresent;
wgpu::RenderPipeline pipelineCopy;

wgpu::Buffer fadeUniformBuffer;
wgpu::Buffer scrollUniformBuffer;
wgpu::Buffer ikedaUniformBuffer;
wgpu::Sampler commonSampler;

wgpu::Texture oldFrameTempTexture;
wgpu::TextureView oldFrameTempView;

wgpu::Texture oldFrameTexture;
wgpu::TextureView oldFrameView;

wgpu::Texture newFrameTexture;
wgpu::TextureView newFrameView;

// ========== Scroll parameters ==========

static float g_offsetX = 0.1f;
static float g_offsetY = 0.0f;
static float g_speedX  = 0.1f;
static float g_speedY  = 0.0f;

// ========== Render parameters (mirrors RenderParams in ikeda_shaders.cpp) ==========

static float g_globalTime  = 0.0f;
static float g_eventPulse  = 0.0f;  // impulse energy, decays every frame
static float g_strobe      = 0.0f;
static float g_sceneMix    = 0.0f;

static float g_styleA       = 2.0f; // BAYER
static float g_styleB       = 8.0f; // HEX
static float g_styleMixProb = 0.15f;
static float g_threshold    = 0.5f;

static float g_ditherScale = 3.0f;
static float g_blockScale  = 14.0f;
static float g_sliceAmp    = 0.15f;
static float g_jitterAmp   = 0.3f;

static float g_moshAmount    = 0.25f;
static float g_moshBlock     = 0.035f;
static float g_moshDrop      = 0.15f;
static float g_feedbackDecay = 0.97f;

static float g_invert      = 0.0f;
static float g_gridOverlay = 0.5f;
static float g_scanline    = 0.35f;
static float g_noiseAmount = 0.08f;

static float g_colorBleed = 0.0f;
static float g_contrast   = 1.4f;

static constexpr float kEventPulseHalfLifeMs = 220.0f;

// ========== Data Structures & decode queue ==========

struct RawArtifact {
    std::vector<uint8_t> bytes;
    uint32_t sequence = 0;
};

struct ImageData {
    std::vector<uint8_t> pixels;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t sequence = 0;
};

template<typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t capacity) : capacity_(capacity) {}

    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) return;
            if (queue_.size() >= capacity_) {
                queue_.pop();
                dropped_++;
            }
            queue_.push(std::move(value));
        }
        condition_.notify_one();
    }

    bool tryPop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool popBlocking(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this]{return closed_ || !queue_.empty();});
        if (queue_.empty()) return false;
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        condition_.notify_all();
    }

    size_t dropped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }

private:
    size_t capacity_;
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable condition_;
    bool closed_ = false;
    size_t dropped_ = 0;
};

ThreadSafeQueue<RawArtifact> rawDataQueue(32);

bool decodeAndResizeImage(const uint8_t* data, int length, uint32_t sequence, ImageData& imageOut) {
    int x, y, n;
    unsigned char* img = stbi_load_from_memory((const unsigned char*)data, length, &x, &y, &n, 4);
    if (!img) {
        std::cerr << "Failed to decode image from memory\n";
        return false;
    }
    const int desiredWidth  = 384;
    const int desiredHeight = 384;
    std::vector<unsigned char> resized(desiredWidth * desiredHeight * 4);
    int result = stbir_resize_uint8(
            img, x, y, 0,
            resized.data(), desiredWidth, desiredHeight, 0, 4
    );
    stbi_image_free(img);
    if (!result) {
        std::cerr << "Failed to resize image.\n";
        return false;
    }
    imageOut.width  = desiredWidth;
    imageOut.height = desiredHeight;
    imageOut.sequence = sequence;
    imageOut.pixels = std::move(resized);
    return true;
}

// ========== Simple LCG-based shuffle to avoid std::shuffle ==========

static unsigned int myRandSeed = 12345u;
static void customShuffle(std::vector<int>& arr) {
    auto nextRand = []() {
        myRandSeed = 1103515245u * myRandSeed + 12345u;
        return (myRandSeed >> 16) & 0x7FFF;
    };
    for (int i = (int)arr.size() - 1; i > 0; i--) {
        int j = nextRand() % (i + 1);
        std::swap(arr[i], arr[j]);
    }
}

// ========== ImageFlasher Class ==========

class ImageFlasher {
public:
    ImageFlasher(wgpu::Device device, uint32_t ringBufferSize, float imageSwitchInterval);
    ~ImageFlasher();

    void pushImage(const ImageData& image);
    void update();
    void renderTiles(wgpu::RenderPassEncoder& pass, int tileFactor);

    // pipeline layout
    wgpu::PipelineLayout getPipelineLayout() const { return pipelineLayout_; }

    // controls the minimum time for each tile to switch
    void setSwitchInterval(float interval) { imageSwitchInterval_ = interval; }

    // stats
    int getBufferUsage() const {
        return imagesInBuffer_;
    }
    int getRingBufferSize() const {
        return ringBufferSize_;
    }

    // decode queue
    ThreadSafeQueue<ImageData>& getImageQueue() { return imageQueue_; }

    // limit how many images we upload from decode queue each frame
    void setMaxUploadsPerFrame(int maxUploads) {
        maxUploadsPerFrame_ = maxUploads;
    }

    // fraction of eligible tiles to actually switch each frame
    void setRandomTileFraction(float frac) {
        randomTileFraction_ = frac;
        std::cout << "[INFO] randomTileFraction_ => " << randomTileFraction_ << "\n";
    }

    // new: set per-frame dt so we can accumulate time for each tile
    void setDeltaTime(float dt) { dt_ = dt; }

private:
    void uploadImage(const ImageData& image);
    uint32_t randomResidentSlot();

    static const uint32_t maxTiles = 65536;

    wgpu::Device device_;
    wgpu::Queue queue_;
    uint32_t ringBufferSize_;

    uint32_t textureWidth_ = 384;
    uint32_t textureHeight_ = 384;
    uint32_t writeIndex_ = 0;
    uint32_t imagesInBuffer_ = 0;

    // the user-provided minimum time between switches
    float imageSwitchInterval_;

    // accumulate times per tile
    float dt_ = 0.0f; // set each frame
    // for each tile: how many ms have passed since last switch
    std::vector<float> tileTimers_;

    wgpu::Sampler sampler_;
    wgpu::Buffer tileStateBuffer_;

    wgpu::PipelineLayout pipelineLayout_;
    wgpu::BindGroupLayout bindGroupLayout_;

    wgpu::Texture textureArray_;
    wgpu::TextureView textureView_;
    wgpu::BindGroup bindGroup_;

    ThreadSafeQueue<ImageData> imageQueue_{32};

    int maxUploadsPerFrame_ = 0;

    // store the ring-buffer index for each tile
    std::vector<uint32_t> tileIndices_;
    std::vector<uint32_t> tileStateData_;
    std::vector<uint32_t> slotSequences_;
    std::unordered_set<uint32_t> presentedSequences_;

    // user wants partial random updates
    float randomTileFraction_ = 0.5f;

};

ImageFlasher* imageFlasher = nullptr;
static int g_tileFactor = 3; // default => 64 tiles

// JS-Exposed
extern "C" {
EMSCRIPTEN_KEEPALIVE
void setTileFactor(int x) {
    if (x < 0) x = 0;
    if (x > 8) x = 8;
    g_tileFactor = x;
    std::cout << "[INFO] setTileFactor => " << g_tileFactor << std::endl;
}

EMSCRIPTEN_KEEPALIVE
void setRandomTileFraction(float fraction) {
    if (imageFlasher) {
        imageFlasher->setRandomTileFraction(fraction);
    }
}
}

// constructor
ImageFlasher::ImageFlasher(wgpu::Device dev, uint32_t ringSize, float switchInterval)
        : device_(dev),
          queue_(dev.GetQueue()),
          ringBufferSize_(ringSize),
          imageSwitchInterval_(switchInterval)
{
    slotSequences_.resize(ringBufferSize_, 0);
    tileStateData_.resize(4 + maxTiles, 0);

    // create sampler
    wgpu::SamplerDescriptor sd = {};
    sd.addressModeU = wgpu::AddressMode::ClampToEdge;
    sd.addressModeV = wgpu::AddressMode::ClampToEdge;
    sd.magFilter    = wgpu::FilterMode::Linear;
    sd.minFilter    = wgpu::FilterMode::Linear;
    sampler_ = device_.CreateSampler(&sd);

    // create a bind group layout
    wgpu::BindGroupLayoutEntry bgle[4] = {};  // Restore 4 bindings for Ikeda shader
    bgle[0].binding = 0;
    bgle[0].visibility = wgpu::ShaderStage::Vertex;
    bgle[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    // Four u32 header fields plus at least one runtime-array element.
    bgle[0].buffer.minBindingSize = 5 * sizeof(uint32_t);

    bgle[1].binding = 1;
    bgle[1].visibility = wgpu::ShaderStage::Fragment;
    bgle[1].texture.sampleType = wgpu::TextureSampleType::Float;
    bgle[1].texture.viewDimension = wgpu::TextureViewDimension::e2DArray;

    bgle[2].binding = 2;
    bgle[2].visibility = wgpu::ShaderStage::Fragment;
    bgle[2].sampler.type = wgpu::SamplerBindingType::Filtering;

    bgle[3].binding = 3;
    bgle[3].visibility = wgpu::ShaderStage::Fragment;
    bgle[3].buffer.type = wgpu::BufferBindingType::Uniform;
    bgle[3].buffer.minBindingSize = 96; // RenderParams struct size

    wgpu::BindGroupLayoutDescriptor bglDesc = {};
    bglDesc.entryCount = 4;  // Restore 4 bindings for Ikeda shader
    bglDesc.entries    = bgle;

    bindGroupLayout_ = device_.CreateBindGroupLayout(&bglDesc);

    wgpu::PipelineLayoutDescriptor pld = {};
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts     = &bindGroupLayout_;
    pipelineLayout_ = device_.CreatePipelineLayout(&pld);

    wgpu::BufferDescriptor tileBufferDescriptor = {};
    tileBufferDescriptor.size = tileStateData_.size() * sizeof(uint32_t);
    tileBufferDescriptor.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    tileStateBuffer_ = device_.CreateBuffer(&tileBufferDescriptor);

    wgpu::TextureDescriptor td = {};
    td.size.width  = textureWidth_;
    td.size.height = textureHeight_;
    td.size.depthOrArrayLayers = ringBufferSize_;
    td.format = wgpu::TextureFormat::RGBA8Unorm;
    td.usage  = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
    textureArray_ = device_.CreateTexture(&td);

    wgpu::TextureViewDescriptor tvd = {};
    tvd.dimension = wgpu::TextureViewDimension::e2DArray;
    tvd.arrayLayerCount = ringBufferSize_;
    textureView_ = textureArray_.CreateView(&tvd);

    wgpu::BindGroupEntry e[4] = {};
    e[0].binding = 0;
    e[0].buffer = tileStateBuffer_;
    e[0].size = tileBufferDescriptor.size;
    e[1].binding = 1;
    e[1].textureView = textureView_;
    e[2].binding = 2;
    e[2].sampler = sampler_;
    e[3].binding = 3;
    e[3].buffer = ikedaUniformBuffer;
    e[3].size = 96;
    wgpu::BindGroupDescriptor bgd = {};
    bgd.layout = bindGroupLayout_;
    bgd.entryCount = 4;
    bgd.entries = e;
    bindGroup_ = device_.CreateBindGroup(&bgd);
}

// destructor
ImageFlasher::~ImageFlasher(){
    std::cout << "ImageFlasher destroyed.\n";
}

void ImageFlasher::pushImage(const ImageData& image){
    imageQueue_.push(image);
}

// uploads one image into ring buffer
void ImageFlasher::uploadImage(const ImageData& image){
    uint32_t idx = writeIndex_;
    if (imagesInBuffer_ < ringBufferSize_) imagesInBuffer_++;

    wgpu::TexelCopyTextureInfo dst = {};
    dst.texture = textureArray_;
    dst.mipLevel = 0;
    dst.origin   = {0, 0, idx};
    dst.aspect   = wgpu::TextureAspect::All;

    wgpu::TexelCopyBufferLayout layout = {};
    layout.offset       = 0;
    layout.bytesPerRow  = image.width * 4;
    layout.rowsPerImage = image.height;

    wgpu::Extent3D extent = {};
    extent.width  = image.width;
    extent.height = image.height;
    extent.depthOrArrayLayers = 1;

    queue_.WriteTexture(&dst, image.pixels.data(), image.pixels.size(), &layout, &extent);
    slotSequences_[idx] = image.sequence;
    writeIndex_ = (writeIndex_ + 1) % ringBufferSize_;
    EM_ASM({ if (Module.onRendererEvent) Module.onRendererEvent(2, $0); }, image.sequence);
}

// Upload a bounded number of decoded images into the single rolling ring.
void ImageFlasher::update(){
    int uploadCount = 0;
    while(true){
        if (maxUploadsPerFrame_ > 0 && uploadCount >= maxUploadsPerFrame_) {
            break;
        }
        ImageData img;
        if(!imageQueue_.tryPop(img)){
            break;
        }
        EM_ASM({ if (Module.onRendererEvent) Module.onRendererEvent(1, $0); }, img.sequence);
        uploadImage(img);
        uploadCount++;
    }
}

uint32_t ImageFlasher::randomResidentSlot() {
    if (imagesInBuffer_ == 0) return 0;
    myRandSeed = 1103515245u * myRandSeed + 12345u;
    if ((myRandSeed & 3u) == 0u) {
        return (writeIndex_ + ringBufferSize_ - 1) % ringBufferSize_;
    }
    return (myRandSeed >> 8u) % imagesInBuffer_;
}

// Update tile state once and render every tile with one instanced draw.
void ImageFlasher::renderTiles(wgpu::RenderPassEncoder& pass, int tileFactor){
    if (imagesInBuffer_ == 0) {
        return;
    }

    int gridSize   = 1 << tileFactor;
    int totalTiles = gridSize * gridSize;

    // ensure tileIndices_ & tileTimers_ have correct size
    if ((int)tileIndices_.size() != totalTiles){
        tileIndices_.resize(totalTiles, (writeIndex_ + ringBufferSize_ - 1) % ringBufferSize_);
        tileTimers_.resize(totalTiles, 0.0f);
        std::cout << "[INFO] tileIndices_ re-init to size " << totalTiles << "\n";
    }

    // ========== 1) Accumulate dt into tileTimers_ ==========
    for (int i=0; i<totalTiles; i++){
        tileTimers_[i] += (dt_ * 0.001f);
    }

    // ========== 2) Build a list of "candidate" tiles whose timers exceed imageSwitchInterval_ ==========
    std::vector<int> candidates;
    candidates.reserve(totalTiles);
    for (int i=0; i<totalTiles; i++){
        if (tileTimers_[i] >= imageSwitchInterval_){
            candidates.push_back(i);
        }
    }

    // ========== 3) We only actually switch a fraction (randomTileFraction_) of these candidates  ==========
    int candidateCount = (int)candidates.size();
    int toSwitch = (int)std::floor(randomTileFraction_ * float(candidateCount));
    if (toSwitch > candidateCount) toSwitch = candidateCount;

    if (toSwitch>0){
        // shuffle the candidate list
        customShuffle(candidates);

        // increment ring-buffer index for the first 'toSwitch' tiles, reset their timer
        for (int i = 0; i < toSwitch; i++){
            int tileId = candidates[i];
            tileIndices_[tileId] = randomResidentSlot();
            tileTimers_[tileId]  = 0.0f; // reset timer
            uint32_t sequence = slotSequences_[tileIndices_[tileId]];
            if (sequence != 0 && presentedSequences_.insert(sequence).second) {
                EM_ASM({ if (Module.onRendererEvent) Module.onRendererEvent(3, $0); }, sequence);
            }
        }
    }
    tileStateData_[0] = static_cast<uint32_t>(gridSize);
    tileStateData_[1] = imagesInBuffer_;
    for (int i = 0; i < totalTiles; ++i) tileStateData_[4 + i] = tileIndices_[i];
    queue_.WriteBuffer(
        tileStateBuffer_, 0, tileStateData_.data(),
        static_cast<size_t>(4 + totalTiles) * sizeof(uint32_t)
    );
    pass.SetBindGroup(0, bindGroup_);
    pass.Draw(6, static_cast<uint32_t>(totalTiles));
}

// ========== Forward declarations for pipeline creation ==========

wgpu::ShaderModule createShaderModule(const char* code);
void createOffscreenTextures(uint32_t w, uint32_t h);
void createPipelineCopy();
void createPipelineImageFlasher();
void createPipelineFade();
void createPipelinePresent();
void updateIkedaUniforms();

// ========== Additional code for the main loop, etc. ==========

static double lastFrameTime = 0.0;
static int droppedFrames = 0;
static int frameCount = 0;


// ========== SCROLLING LOGIC ==========

void updateScrolling(double dt) {
    if (!scrollUniformBuffer) return;
    g_offsetX += float(g_speedX * (dt * 0.001));
    g_offsetY += float(g_speedY * (dt * 0.001));

    // wrap in [0..1)
    g_offsetX = std::fmod(g_offsetX, 1.0f);
    if (g_offsetX < 0.f) g_offsetX += 1.f;
    g_offsetY = std::fmod(g_offsetY, 1.0f);
    if (g_offsetY < 0.f) g_offsetY += 1.f;

    float data[2] = {g_offsetX, g_offsetY};
    queue.WriteBuffer(scrollUniformBuffer, 0, data, sizeof(data));
}


extern "C" void initializeSurfaceAndPipeline() {
    double cw, ch;
    emscripten_get_element_css_size("canvas", &cw, &ch);
    g_canvasWidth  = std::max<uint32_t>(1, static_cast<uint32_t>(cw));
    g_canvasHeight = std::max<uint32_t>(1, static_cast<uint32_t>(ch));

    wgpu::SurfaceCapabilities capabilities{};
    surfaceGlobal.GetCapabilities(adapter, &capabilities);
    if (capabilities.formatCount == 0) {
        std::cerr << "Surface reported no supported formats.\n";
        return;
    }
    swapChainFormat = capabilities.formats[0];

    wgpu::SurfaceConfiguration surfaceConfig{};
    surfaceConfig.device = device;
    surfaceConfig.format = swapChainFormat;
    surfaceConfig.usage = wgpu::TextureUsage::RenderAttachment;
    surfaceConfig.width = g_canvasWidth;
    surfaceConfig.height = g_canvasHeight;
    surfaceConfig.alphaMode = wgpu::CompositeAlphaMode::Auto;
    surfaceConfig.presentMode = wgpu::PresentMode::Fifo;
    surfaceGlobal.Configure(&surfaceConfig);

    // Create ikedaUniformBuffer before ImageFlasher constructor
    {
        wgpu::BufferDescriptor bd = {};
        bd.size  = 24 * sizeof(float); // RenderParams: 24 floats = 96 bytes
        bd.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        ikedaUniformBuffer = device.CreateBuffer(&bd);
    }

    // create the ImageFlasher
    imageFlasher = new ImageFlasher(device, 256, /*imageSwitchInterval=*/1.0f/3);

    createPipelineImageFlasher();
    createPipelineFade();
    createPipelinePresent();
    createPipelineCopy();
    createOffscreenTextures(g_canvasWidth, g_canvasHeight);

    // a sampler for passes like copy/present
    {
        wgpu::SamplerDescriptor sd = {};
        sd.minFilter       = wgpu::FilterMode::Linear;
        sd.magFilter       = wgpu::FilterMode::Linear;
        sd.addressModeU    = wgpu::AddressMode::ClampToEdge;
        sd.addressModeV    = wgpu::AddressMode::ClampToEdge;
        commonSampler      = device.CreateSampler(&sd);
    }

    // main render loop
    emscripten_request_animation_frame_loop([](double time, void*) {
        double dt = (lastFrameTime > 0) ? (time - lastFrameTime) : 0.0;
        lastFrameTime = time;
        frameCount++;
        if (dt > 25.0) {
            droppedFrames++;
        }
        if (frameCount % 60 == 0) {
            std::cout << "[DEBUG] Frame: " << frameCount
                      << " | Dropped: " << droppedFrames << std::endl;
        }

        // pass dt in ms to the flasher
        if (imageFlasher) {
            imageFlasher->setDeltaTime((float)dt);
        }

        // advance global time and decay event energy toward zero
        g_globalTime = (float)(time * 0.001); // convert to seconds
        if (dt > 0.0) {
            g_eventPulse *= std::exp2(-(float)dt / kEventPulseHalfLifeMs);
            if (g_eventPulse < 0.001f) g_eventPulse = 0.0f;
        }
        updateIkedaUniforms();

        // scroll offset
        updateScrolling(dt);

        wgpu::SurfaceTexture surfaceTexture{};
        surfaceGlobal.GetCurrentTexture(&surfaceTexture);
        wgpu::TextureView swapChainView = surfaceTexture.texture.CreateView();
        if (!swapChainView) return EM_TRUE;

        // update ring buffer from decode queue, etc
        imageFlasher->update();

        wgpu::CommandEncoder encoder = device.CreateCommandEncoder({});

        // Pass #1: newFrame
        {
            wgpu::RenderPassColorAttachment att = {};
            att.view       = newFrameView;
            att.loadOp     = wgpu::LoadOp::Clear;
            att.storeOp    = wgpu::StoreOp::Store;
            att.clearValue = {0,0,0,1};

            wgpu::RenderPassDescriptor desc = {};
            desc.colorAttachmentCount = 1;
            desc.colorAttachments     = &att;

            wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&desc);
            pass.SetPipeline(pipelineImageFlasher);

            // do the tile-based draws, with partial random updates
            imageFlasher->renderTiles(pass, g_tileFactor);

            pass.End();
        }

        // Pass #2: copy oldFrame => oldFrameTemp
        {
            wgpu::BindGroup copyBG = [&]{
                wgpu::BindGroupLayout bgl = pipelineCopy.GetBindGroupLayout(0);
                wgpu::BindGroupEntry entries[2] = {};
                entries[0].binding     = 0;
                entries[0].textureView = oldFrameView;
                entries[1].binding     = 1;
                entries[1].sampler     = commonSampler;

                wgpu::BindGroupDescriptor bd = {};
                bd.layout     = bgl;
                bd.entryCount = 2;
                bd.entries    = entries;
                return device.CreateBindGroup(&bd);
            }();

            wgpu::RenderPassColorAttachment att = {};
            att.view       = oldFrameTempView;
            att.loadOp     = wgpu::LoadOp::Clear;
            att.storeOp    = wgpu::StoreOp::Store;
            att.clearValue = {0,0,0,1};

            wgpu::RenderPassDescriptor desc = {};
            desc.colorAttachmentCount = 1;
            desc.colorAttachments     = &att;

            wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&desc);
            pass.SetPipeline(pipelineCopy);
            pass.SetBindGroup(0, copyBG);
            pass.Draw(6);
            pass.End();
        }

        // Pass #3: fade => oldFrame
        {
            wgpu::BindGroup fadeBG = [&]{
                wgpu::BindGroupLayout bgl = pipelineFade.GetBindGroupLayout(0);
                wgpu::BindGroupEntry e[5] = {};
                e[0].binding     = 0; // oldFrame
                e[0].textureView = oldFrameTempView;
                e[1].binding     = 1; // newFrame
                e[1].textureView = newFrameView;
                e[2].binding     = 2; // fadeUniform
                e[2].buffer      = fadeUniformBuffer;
                e[2].size        = sizeof(float);
                e[3].binding     = 3; // sampler
                e[3].sampler     = commonSampler;
                e[4].binding     = 4; // ikeda uniform
                e[4].buffer      = ikedaUniformBuffer;
                e[4].size        = 96; // RenderParams struct size

                wgpu::BindGroupDescriptor bd = {};
                bd.layout     = bgl;
                bd.entryCount = 5;
                bd.entries    = e;
                return device.CreateBindGroup(&bd);
            }();

            wgpu::RenderPassColorAttachment att = {};
            att.view       = oldFrameView;
            att.loadOp     = wgpu::LoadOp::Clear;
            att.storeOp    = wgpu::StoreOp::Store;
            att.clearValue = {0,0,0,1};

            wgpu::RenderPassDescriptor desc = {};
            desc.colorAttachmentCount = 1;
            desc.colorAttachments     = &att;

            wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&desc);
            pass.SetPipeline(pipelineFade);
            pass.SetBindGroup(0, fadeBG);
            pass.Draw(6);
            pass.End();
        }

        // Pass #4: present => swapChain
        {
            wgpu::BindGroup presentBG = [&] {
                wgpu::BindGroupLayout bgl = pipelinePresent.GetBindGroupLayout(0);
                wgpu::BindGroupEntry e[4] = {};
                e[0].binding      = 0;
                e[0].textureView  = oldFrameView;
                e[1].binding      = 1;
                e[1].sampler      = commonSampler;
                e[2].binding      = 2;
                e[2].buffer       = scrollUniformBuffer;
                e[2].size         = 2*sizeof(float);
                e[3].binding      = 3;
                e[3].buffer       = ikedaUniformBuffer;
                e[3].size         = 96; // RenderParams struct size

                wgpu::BindGroupDescriptor bd = {};
                bd.layout     = bgl;
                bd.entryCount = 4;
                bd.entries    = e;
                return device.CreateBindGroup(&bd);
            }();

            wgpu::RenderPassColorAttachment att = {};
            att.view       = swapChainView;
            att.loadOp     = wgpu::LoadOp::Clear;
            att.storeOp    = wgpu::StoreOp::Store;
            att.clearValue = {0.3f, 0.3f, 0.3f, 1.0f};

            wgpu::RenderPassDescriptor desc = {};
            desc.colorAttachmentCount = 1;
            desc.colorAttachments     = &att;

            wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&desc);
            pass.SetPipeline(pipelinePresent);
            pass.SetBindGroup(0, presentBG);
            pass.Draw(6);
            pass.End();
        }

        // submit
        wgpu::CommandBuffer cmd = encoder.Finish();
        queue.Submit(1, &cmd);

        return EM_TRUE;
    }, nullptr);
}

// decode worker
std::atomic<bool> decodeWorkerRunning(true);
std::thread decodeWorkerThread;

void decodeWorkerFunc() {
    while (decodeWorkerRunning) {
        RawArtifact rawData;
        if (!rawDataQueue.popBlocking(rawData)) {
            break;
        }
        ImageData imgData;
        if (!decodeAndResizeImage(
                rawData.bytes.data(),
                static_cast<int>(rawData.bytes.size()),
                rawData.sequence,
                imgData)) {
            EM_ASM({ if (Module.onRendererEvent) Module.onRendererEvent(4, $0); }, rawData.sequence);
            continue;
        }
        if (imageFlasher) {
            imageFlasher->pushImage(imgData);
        }
    }
}

// Upload the shared RenderParams uniform (must match ikeda_shaders.cpp layout).
void updateIkedaUniforms() {
    if (!ikedaUniformBuffer) return;

    struct RenderParams {
        float time, eventPulse, strobe, sceneMix;
        float styleA, styleB, styleMixProb, threshold;
        float ditherScale, blockScale, sliceAmp, jitterAmp;
        float moshAmount, moshBlock, moshDrop, feedbackDecay;
        float invert, gridOverlay, scanline, noiseAmount;
        float canvasWidth, canvasHeight, colorBleed, contrast;
    } p;
    static_assert(sizeof(RenderParams) == 96, "RenderParams must stay 96 bytes");

    p.time = g_globalTime;
    p.eventPulse = g_eventPulse;
    p.strobe = g_strobe;
    p.sceneMix = g_sceneMix;
    p.styleA = g_styleA;
    p.styleB = g_styleB;
    p.styleMixProb = g_styleMixProb;
    p.threshold = g_threshold;
    p.ditherScale = g_ditherScale;
    p.blockScale = g_blockScale;
    p.sliceAmp = g_sliceAmp;
    p.jitterAmp = g_jitterAmp;
    p.moshAmount = g_moshAmount;
    p.moshBlock = g_moshBlock;
    p.moshDrop = g_moshDrop;
    p.feedbackDecay = g_feedbackDecay;
    p.invert = g_invert;
    p.gridOverlay = g_gridOverlay;
    p.scanline = g_scanline;
    p.noiseAmount = g_noiseAmount;
    p.canvasWidth = (float)g_canvasWidth;
    p.canvasHeight = (float)g_canvasHeight;
    p.colorBleed = g_colorBleed;
    p.contrast = g_contrast;

    queue.WriteBuffer(ikedaUniformBuffer, 0, &p, sizeof(p));
}

// EMSCRIPTEN exports
extern "C" {
EMSCRIPTEN_KEEPALIVE
void onArtifactReceived(uint8_t* data, int length, uint32_t sequence) {
    RawArtifact artifact;
    artifact.bytes.assign(data, data + length);
    artifact.sequence = sequence;
    rawDataQueue.push(std::move(artifact));
}

EMSCRIPTEN_KEEPALIVE
void onImageReceived(uint8_t* data, int length) {
    onArtifactReceived(data, length, 0);
}

EMSCRIPTEN_KEEPALIVE
void setFadeFactor(float factor) {
    if (!fadeUniformBuffer) return;
    queue.WriteBuffer(fadeUniformBuffer, 0, &factor, sizeof(float));
}

EMSCRIPTEN_KEEPALIVE
void setImageSwitchInterval(float interval) {
    if (imageFlasher) {
        imageFlasher->setSwitchInterval(interval);
    }
}

EMSCRIPTEN_KEEPALIVE
int getBufferUsage() {
    if (!imageFlasher) return 0;
    return imageFlasher->getBufferUsage();
}

EMSCRIPTEN_KEEPALIVE
int getRingBufferSize() {
    if (!imageFlasher) return 0;
    return imageFlasher->getRingBufferSize();
}

EMSCRIPTEN_KEEPALIVE
void setMaxUploadsPerFrame(int maxUploads) {
    if (imageFlasher) {
        imageFlasher->setMaxUploadsPerFrame(maxUploads);
    }
}

// set scrolling speed
EMSCRIPTEN_KEEPALIVE
void setScrollingSpeed(float sx, float sy) {
    g_speedX = sx;
    g_speedY = sy;
    std::cout << "[INFO] setScrollingSpeed(" << sx << ", " << sy << ")\n";
}

// set immediate scrolling offset
EMSCRIPTEN_KEEPALIVE
void setScrollingOffset(float ox, float oy) {
    g_offsetX = ox - std::floor(ox);
    g_offsetY = oy - std::floor(oy);
    if (scrollUniformBuffer) {
        float data[2] = {g_offsetX, g_offsetY};
        queue.WriteBuffer(scrollUniformBuffer, 0, data, sizeof(data));
    }
    std::cout << "[INFO] setScrollingOffset(" << ox << ", " << oy << ")\n";
}

// ========== Render parameter API (called from app.js) ==========

// Tile materials: 0 RAW, 1 THRESH, 2 BAYER, 3 EDGE, 4 SORT, 5 SLICE,
//                 6 WAVE, 7 BARCODE, 8 HEX, 9 BLOCKS
EMSCRIPTEN_KEEPALIVE
void setStyles(int styleA, int styleB, float mixProb) {
    g_styleA = (float)std::clamp(styleA, 0, 9);
    g_styleB = (float)std::clamp(styleB, 0, 9);
    g_styleMixProb = std::clamp(mixProb, 0.0f, 1.0f);
}

EMSCRIPTEN_KEEPALIVE
void setTone(float threshold, float contrast, float colorBleed, float jitter) {
    g_threshold  = std::clamp(threshold, 0.0f, 1.0f);
    g_contrast   = std::clamp(contrast, 0.1f, 4.0f);
    g_colorBleed = std::clamp(colorBleed, 0.0f, 1.0f);
    g_jitterAmp  = std::clamp(jitter, 0.0f, 1.0f);
}

EMSCRIPTEN_KEEPALIVE
void setStructure(float ditherScale, float blockScale, float sliceAmp, float grid) {
    g_ditherScale = std::clamp(ditherScale, 1.0f, 16.0f);
    g_blockScale  = std::clamp(blockScale, 6.0f, 48.0f);
    g_sliceAmp    = std::clamp(sliceAmp, 0.0f, 1.0f);
    g_gridOverlay = std::clamp(grid, 0.0f, 1.0f);
}

EMSCRIPTEN_KEEPALIVE
void setMosh(float amount, float block, float drop, float decay) {
    g_moshAmount    = std::clamp(amount, 0.0f, 1.0f);
    g_moshBlock     = std::clamp(block, 0.004f, 0.25f);
    g_moshDrop      = std::clamp(drop, 0.0f, 1.0f);
    g_feedbackDecay = std::clamp(decay, 0.5f, 1.0f);
}

EMSCRIPTEN_KEEPALIVE
void setTemporal(float scanline, float noise, float strobe, float invert) {
    g_scanline    = std::clamp(scanline, 0.0f, 1.0f);
    g_noiseAmount = std::clamp(noise, 0.0f, 1.0f);
    g_strobe      = std::clamp(strobe, 0.0f, 1.0f);
    g_invert      = std::clamp(invert, 0.0f, 1.0f);
}

// Inject event energy: spikes mosh, slice glitch, and tile negatives, then decays.
EMSCRIPTEN_KEEPALIVE
void pulse(float strength) {
    g_eventPulse = std::clamp(g_eventPulse + strength, 0.0f, 1.5f);
}

} // extern "C"

void cleanup() {
    decodeWorkerRunning = false;
    rawDataQueue.close();
    if (decodeWorkerThread.joinable()) {
        decodeWorkerThread.join();
    }
    delete imageFlasher;
    imageFlasher = nullptr;
}

int main() {
    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvasSource{};
    canvasSource.selector = "#canvas";

    wgpu::SurfaceDescriptor surfaceDescriptor{};
    surfaceDescriptor.nextInChain = &canvasSource;
    surfaceGlobal = instance.CreateSurface(&surfaceDescriptor);
    if (!surfaceGlobal) {
        std::cerr << "Failed to create surface.\n";
        return -1;
    }

    wgpu::RequestAdapterOptions options{};
    options.compatibleSurface = surfaceGlobal;
    options.powerPreference = wgpu::PowerPreference::HighPerformance;

    instance.RequestAdapter(
        &options,
        wgpu::CallbackMode::AllowSpontaneous,
        [](wgpu::RequestAdapterStatus status, wgpu::Adapter requestedAdapter,
           wgpu::StringView message) {
            if (status != wgpu::RequestAdapterStatus::Success) {
                std::cerr << "Failed to get WebGPU adapter: ";
                if (message.length) std::cerr.write(message.data, message.length);
                std::cerr << std::endl;
                return;
            }

            adapter = requestedAdapter;
            wgpu::DeviceDescriptor deviceDescriptor{};
            deviceDescriptor.label = "ImageFlasher renderer";
            deviceDescriptor.SetUncapturedErrorCallback(
                [](const wgpu::Device&, wgpu::ErrorType type, wgpu::StringView error) {
                    std::string message(error.data ? error.data : "", error.length);
                    std::cerr << "Uncaptured WebGPU error ("
                              << static_cast<int>(type) << "): ";
                    if (!message.empty()) std::cerr << message;
                    std::cerr << std::endl;
                    EM_ASM({
                        if (Module.onWebGPUError) {
                            Module.onWebGPUError($0, UTF8ToString($1));
                        }
                    }, static_cast<int>(type), message.c_str());
                });

            adapter.RequestDevice(
                &deviceDescriptor,
                wgpu::CallbackMode::AllowSpontaneous,
                [](wgpu::RequestDeviceStatus deviceStatus, wgpu::Device requestedDevice,
                   wgpu::StringView deviceMessage) {
                    if (deviceStatus != wgpu::RequestDeviceStatus::Success) {
                        std::cerr << "Failed to create WebGPU device: ";
                        if (deviceMessage.length) {
                            std::cerr.write(deviceMessage.data, deviceMessage.length);
                        }
                        std::cerr << std::endl;
                        return;
                    }

                    device = requestedDevice;
                    queue = device.GetQueue();
                    initializeSurfaceAndPipeline();
                    decodeWorkerThread = std::thread(decodeWorkerFunc);
                });
        });

    return 0;
}

// ========== Implementation of Pipeline Helpers ==========

wgpu::ShaderModule createShaderModule(const char* code) {
    wgpu::ShaderSourceWGSL wgslDesc = {};
    wgslDesc.code = code;

    wgpu::ShaderModuleDescriptor desc = {};
    desc.nextInChain = &wgslDesc;
    desc.label = "Shader Module";

    return device.CreateShaderModule(&desc);
}

void createOffscreenTextures(uint32_t w, uint32_t h) {
    {
        wgpu::TextureDescriptor desc = {};
        desc.size.width  = w;
        desc.size.height = h;
        desc.size.depthOrArrayLayers = 1;
        desc.format = wgpu::TextureFormat::RGBA8Unorm;
        desc.usage  = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
        oldFrameTexture = device.CreateTexture(&desc);
        oldFrameView    = oldFrameTexture.CreateView();
    }
    {
        wgpu::TextureDescriptor desc = {};
        desc.size.width  = w;
        desc.size.height = h;
        desc.size.depthOrArrayLayers = 1;
        desc.format = wgpu::TextureFormat::RGBA8Unorm;
        desc.usage  = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
        newFrameTexture = device.CreateTexture(&desc);
        newFrameView    = newFrameTexture.CreateView();
    }
    {
        wgpu::TextureDescriptor desc = {};
        desc.size.width  = w;
        desc.size.height = h;
        desc.size.depthOrArrayLayers = 1;
        desc.format = wgpu::TextureFormat::RGBA8Unorm;
        desc.usage  = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
        oldFrameTempTexture = device.CreateTexture(&desc);
        oldFrameTempView    = oldFrameTempTexture.CreateView();
    }
}

void createPipelineCopy() {
    wgpu::ShaderModule vs = createShaderModule(vertexShaderWGSL);
    wgpu::ShaderModule fs = createShaderModule(copyFragmentWGSL);

    wgpu::BindGroupLayoutEntry bglEntries[2] = {};
    bglEntries[0].binding    = 0;
    bglEntries[0].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[0].texture.sampleType    = wgpu::TextureSampleType::Float;
    bglEntries[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;

    bglEntries[1].binding    = 1;
    bglEntries[1].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[1].sampler.type = wgpu::SamplerBindingType::Filtering;

    wgpu::BindGroupLayoutDescriptor bglDesc = {};
    bglDesc.entryCount = 2;
    bglDesc.entries    = bglEntries;
    wgpu::BindGroupLayout copyBGL = device.CreateBindGroupLayout(&bglDesc);

    wgpu::PipelineLayoutDescriptor pld = {};
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts     = &copyBGL;
    wgpu::PipelineLayout layout = device.CreatePipelineLayout(&pld);

    wgpu::RenderPipelineDescriptor desc = {};
    desc.layout              = layout;
    desc.vertex.module       = vs;
    desc.vertex.entryPoint   = "vsMain";

    wgpu::ColorTargetState ct = {};
    ct.format    = wgpu::TextureFormat::RGBA8Unorm;
    ct.writeMask = wgpu::ColorWriteMask::All;

    wgpu::FragmentState fsState = {};
    fsState.module      = fs;
    fsState.entryPoint  = "fsCopy";
    fsState.targetCount = 1;
    fsState.targets     = &ct;
    desc.fragment       = &fsState;

    desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    desc.primitive.cullMode = wgpu::CullMode::None;
    desc.multisample.count  = 1;

    pipelineCopy = device.CreateRenderPipeline(&desc);
}

void createPipelineImageFlasher() {
    wgpu::ShaderModule vs = createShaderModule(tileVertexShaderWGSL);
    wgpu::ShaderModule fs = createShaderModule(ikedaImageFlasherFragmentWGSL);
    wgpu::PipelineLayout layout = imageFlasher->getPipelineLayout();

    wgpu::RenderPipelineDescriptor desc = {};
    desc.layout              = layout;
    desc.vertex.module       = vs;
    desc.vertex.entryPoint   = "vsTile";

    wgpu::ColorTargetState colorTarget = {};
    colorTarget.format    = wgpu::TextureFormat::RGBA8Unorm;
    colorTarget.writeMask = wgpu::ColorWriteMask::All;

    wgpu::FragmentState fsState = {};
    fsState.module      = fs;
    fsState.entryPoint  = "fsImage";
    fsState.targetCount = 1;
    fsState.targets     = &colorTarget;
    desc.fragment       = &fsState;

    desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    desc.primitive.cullMode = wgpu::CullMode::None;
    desc.multisample.count  = 1;

    pipelineImageFlasher = device.CreateRenderPipeline(&desc);
}

void createPipelineFade() {
    wgpu::ShaderModule vs = createShaderModule(vertexShaderWGSL);
    wgpu::ShaderModule fs = createShaderModule(ikedaFadeFragmentWGSL);

    wgpu::BindGroupLayoutEntry bglEntries[5] = {};
    bglEntries[0].binding    = 0;
    bglEntries[0].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[0].texture.sampleType    = wgpu::TextureSampleType::Float;
    bglEntries[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;

    bglEntries[1].binding    = 1;
    bglEntries[1].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[1].texture.sampleType    = wgpu::TextureSampleType::Float;
    bglEntries[1].texture.viewDimension = wgpu::TextureViewDimension::e2D;

    bglEntries[2].binding    = 2;
    bglEntries[2].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[2].buffer.type= wgpu::BufferBindingType::Uniform;

    bglEntries[3].binding    = 3;
    bglEntries[3].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[3].sampler.type = wgpu::SamplerBindingType::Filtering;

    bglEntries[4].binding    = 4;
    bglEntries[4].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[4].buffer.type= wgpu::BufferBindingType::Uniform;
    bglEntries[4].buffer.minBindingSize = 96; // RenderParams struct size

    wgpu::BindGroupLayoutDescriptor bglDesc = {};
    bglDesc.entryCount = 5;
    bglDesc.entries    = bglEntries;
    wgpu::BindGroupLayout fadeBGL = device.CreateBindGroupLayout(&bglDesc);

    wgpu::PipelineLayoutDescriptor plDesc = {};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts     = &fadeBGL;
    wgpu::PipelineLayout layout = device.CreatePipelineLayout(&plDesc);

    wgpu::RenderPipelineDescriptor descP = {};
    descP.layout              = layout;
    descP.vertex.module       = vs;
    descP.vertex.entryPoint   = "vsMain";

    wgpu::ColorTargetState colorTarget = {};
    colorTarget.format    = wgpu::TextureFormat::RGBA8Unorm;
    colorTarget.writeMask = wgpu::ColorWriteMask::All;

    wgpu::FragmentState fsState = {};
    fsState.module       = fs;
    fsState.entryPoint   = "fsFade";
    fsState.targetCount  = 1;
    fsState.targets      = &colorTarget;
    descP.fragment       = &fsState;

    descP.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    descP.primitive.cullMode = wgpu::CullMode::None;
    descP.multisample.count  = 1;

    pipelineFade = device.CreateRenderPipeline(&descP);

    // fadeUniformBuffer
    {
        wgpu::BufferDescriptor bd = {};
        bd.size  = sizeof(float);
        bd.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        fadeUniformBuffer = device.CreateBuffer(&bd);

        float fadeFactor = 0.5f;
        queue.WriteBuffer(fadeUniformBuffer, 0, &fadeFactor, sizeof(fadeFactor));
    }
}

void createPipelinePresent() {
    wgpu::ShaderModule vs = createShaderModule(vertexShaderWGSL);
    wgpu::ShaderModule fs = createShaderModule(ikedaPresentFragmentWGSL);

    wgpu::BindGroupLayoutEntry bglEntries[4] = {};
    // oldFrame
    bglEntries[0].binding    = 0;
    bglEntries[0].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[0].texture.sampleType     = wgpu::TextureSampleType::Float;
    bglEntries[0].texture.viewDimension  = wgpu::TextureViewDimension::e2D;

    // sampler
    bglEntries[1].binding    = 1;
    bglEntries[1].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[1].sampler.type = wgpu::SamplerBindingType::Filtering;

    // scrollParam
    bglEntries[2].binding    = 2;
    bglEntries[2].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[2].buffer.type= wgpu::BufferBindingType::Uniform;

    // ikeda uniform
    bglEntries[3].binding    = 3;
    bglEntries[3].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[3].buffer.type= wgpu::BufferBindingType::Uniform;
    bglEntries[3].buffer.minBindingSize = 96; // RenderParams struct size

    wgpu::BindGroupLayoutDescriptor bglDesc = {};
    bglDesc.entryCount = 4;
    bglDesc.entries    = bglEntries;
    wgpu::BindGroupLayout presentBGL = device.CreateBindGroupLayout(&bglDesc);

    wgpu::PipelineLayoutDescriptor plDesc = {};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts     = &presentBGL;
    wgpu::PipelineLayout layout = device.CreatePipelineLayout(&plDesc);

    wgpu::RenderPipelineDescriptor descP = {};
    descP.layout              = layout;
    descP.vertex.module       = vs;
    descP.vertex.entryPoint   = "vsMain";

    wgpu::ColorTargetState ct = {};
    ct.format    = swapChainFormat;
    ct.writeMask = wgpu::ColorWriteMask::All;

    wgpu::FragmentState fsState = {};
    fsState.module      = fs;
    fsState.entryPoint  = "fsPresent";
    fsState.targetCount = 1;
    fsState.targets     = &ct;
    descP.fragment      = &fsState;

    descP.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    descP.primitive.cullMode = wgpu::CullMode::None;
    descP.multisample.count  = 1;

    pipelinePresent = device.CreateRenderPipeline(&descP);

    // create scrollUniformBuffer
    {
        wgpu::BufferDescriptor bd = {};
        bd.size  = 2 * sizeof(float);
        bd.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        scrollUniformBuffer = device.CreateBuffer(&bd);

        float init[2] = {0.0f, 0.0f};
        queue.WriteBuffer(scrollUniformBuffer, 0, init, sizeof(init));
    }

    // Initialize ikedaUniformBuffer with default values (buffer created earlier)
    updateIkedaUniforms();
}
