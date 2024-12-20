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
#include <filesystem>  // C++17 needed
#include <set>
#include <deque>

#include <emscripten.h>
#include <emscripten/html5.h> // For emscripten_request_animation_frame_loop
#include <emscripten/html5_webgpu.h>

#include <webgpu/webgpu_cpp.h>

// Include stb_image and stb_image_resize for loading and resizing PNG images
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

// Error handler function
void HandleUncapturedError(WGPUErrorType type, const char* message, void* userdata) {
    emscripten_log(EM_LOG_ERROR, "Uncaptured WebGPU Error (%d): %s", static_cast<int>(type), message);
}

// Vertex Shader
const char* vertexShaderCode = R"(
struct VertexOut {
    @builtin(position) Position : vec4<f32>,
    @location(0) fragUV : vec2<f32>,
};

@vertex
fn main(@builtin(vertex_index) VertexIndex : u32) -> VertexOut {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0)
    );
    var uv = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 0.0)
    );
    var output : VertexOut;
    output.Position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
    output.fragUV = uv[VertexIndex];
    return output;
}
)";

// Fragment Shader
const char* fragmentShaderCode = R"(
struct Uniforms {
    layerIndex : i32
}
@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var textureArray : texture_2d_array<f32>;
@group(0) @binding(2) var sampler0 : sampler;

@fragment
fn main(@location(0) fragUV : vec2<f32>) -> @location(0) vec4<f32> {
    let color = textureSample(textureArray, sampler0, fragUV, uniforms.layerIndex);
    return color;
}
)";

// Global variables for device and rendering
wgpu::Device device;
wgpu::Queue queue;
wgpu::SwapChain swapChain;
wgpu::RenderPipeline pipeline;
wgpu::Surface surfaceGlobal;
wgpu::TextureFormat swapChainFormat;

// Forward declarations
EM_BOOL frame(double time, void* userData);
void cleanup();

// Image data structure
struct ImageData {
    std::vector<uint8_t> pixels; // RGBA8 format
    uint32_t width;
    uint32_t height;
};

// Thread-safe queue implementation
template<typename T>
class ThreadSafeQueue {
public:
    void push(const T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(value);
        condition_.notify_one();
    }

    bool tryPop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty())
            return false;
        value = queue_.front();
        queue_.pop();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable condition_;
};

// Global variables for image generation control
std::atomic<bool> imageGenRunning(false);
std::thread imageGenThread;

// ImageFlasher class definition with Double Buffering and Time-based switching
class ImageFlasher {
public:
    ImageFlasher(wgpu::Device device, uint32_t ringBufferSize, float imageSwitchInterval);
    ~ImageFlasher();

    void pushImage(const ImageData& image);
    void update(); // Call in main loop
    void render(wgpu::RenderPassEncoder& pass);
    void swapBuffers(); // Swap front and back buffers

    wgpu::PipelineLayout getPipelineLayout() const { return pipelineLayout_; }

private:
    void uploadImage(const ImageData& image, int buffer);

    static const uint32_t maxLayersPerArray = 256;

    wgpu::Device device_;
    wgpu::Queue queue_;

    uint32_t ringBufferSize_;
    uint32_t textureWidth_ = 512;
    uint32_t textureHeight_ = 512;

    // Each buffer (0 or 1) has its own ring state
    uint32_t writeIndex_[2];
    uint32_t displayIndex_[2];
    uint32_t imagesInBuffer_[2];

    // For time-based switching
    float imageSwitchInterval_;
    std::chrono::steady_clock::time_point lastSwitchTime_[2];

    wgpu::Sampler sampler_;
    std::array<wgpu::Buffer, 2> uniformBuffers_;
    wgpu::PipelineLayout pipelineLayout_;
    wgpu::BindGroupLayout bindGroupLayout_;

    std::array<std::vector<wgpu::Texture>, 2> textureArrays_;
    std::array<std::vector<wgpu::TextureView>, 2> textureViews_;
    std::array<std::vector<wgpu::BindGroup>, 2> bindGroups_;

    mutable std::mutex mutex_;
    ThreadSafeQueue<ImageData> imageQueue_;
    int bufferIndex_; // 0 or 1
};

ImageFlasher::ImageFlasher(wgpu::Device device, uint32_t ringBufferSize, float imageSwitchInterval)
        : device_(device),
          queue_(device.GetQueue()),
          ringBufferSize_(ringBufferSize),
          bufferIndex_(0),
          imageSwitchInterval_(imageSwitchInterval) {
    std::cout << "Initializing ImageFlasher with ring buffer size: " << ringBufferSize_ << " and interval: " << imageSwitchInterval_ << std::endl;

    for (int b = 0; b < 2; ++b) {
        writeIndex_[b] = 0;
        displayIndex_[b] = 0;
        imagesInBuffer_[b] = 0;
        lastSwitchTime_[b] = std::chrono::steady_clock::now();
    }

    uint32_t numTextureArrays = (ringBufferSize_ + maxLayersPerArray - 1) / maxLayersPerArray;
    std::cout << "Number of texture arrays: " << numTextureArrays << std::endl;

    for (int b = 0; b < 2; ++b) {
        textureArrays_[b].resize(numTextureArrays);
        textureViews_[b].resize(numTextureArrays);
        bindGroups_[b].resize(numTextureArrays);
    }

    wgpu::SamplerDescriptor samplerDesc = {};
    samplerDesc.addressModeU = wgpu::AddressMode::ClampToEdge;
    samplerDesc.addressModeV = wgpu::AddressMode::ClampToEdge;
    samplerDesc.magFilter = wgpu::FilterMode::Linear;
    samplerDesc.minFilter = wgpu::FilterMode::Linear;
    sampler_ = device_.CreateSampler(&samplerDesc);
    std::cout << "Sampler created." << std::endl;

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
    std::cout << "Bind group layout created." << std::endl;

    wgpu::PipelineLayoutDescriptor plDesc = {};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &bindGroupLayout_;
    pipelineLayout_ = device_.CreatePipelineLayout(&plDesc);
    std::cout << "Pipeline layout created." << std::endl;

    for (int b = 0; b < 2; ++b) {
        wgpu::BufferDescriptor uniformBufferDesc = {};
        uniformBufferDesc.size = 16;
        uniformBufferDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        uniformBuffers_[b] = device_.CreateBuffer(&uniformBufferDesc);
        std::cout << "Uniform buffer " << b << " created." << std::endl;

        for (uint32_t i = 0; i < numTextureArrays; ++i) {
            uint32_t layersInThisArray = maxLayersPerArray;
            if (i == numTextureArrays -1 && (ringBufferSize_ % maxLayersPerArray != 0)) {
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
            std::cout << "Buffer " << b << ": Texture array " << i << " created with " << layersInThisArray << " layers." << std::endl;

            wgpu::TextureViewDescriptor viewDesc = {};
            viewDesc.dimension = wgpu::TextureViewDimension::e2DArray;
            textureViews_[b][i] = textureArray.CreateView(&viewDesc);
            std::cout << "Buffer " << b << ": Texture view " << i << " created." << std::endl;

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
            std::cout << "Buffer " << b << ": Bind group " << i << " created." << std::endl;
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
        displayIndex_[buffer] = (displayIndex_[buffer] + 1) % ringBufferSize_;
    } else {
        imagesInBuffer_[buffer]++;
    }

    uint32_t globalLayerIndex = writeIndex_[buffer];
    uint32_t textureArrayIndex = globalLayerIndex / maxLayersPerArray;
    uint32_t layerIndexInTexture = globalLayerIndex % maxLayersPerArray;

    std::cout << "Buffer " << buffer << ": Uploading image to texture array " << textureArrayIndex << ", layer " << layerIndexInTexture << std::endl;

    wgpu::ImageCopyTexture dst = {};
    dst.texture = textureArrays_[buffer][textureArrayIndex];
    dst.mipLevel = 0;
    dst.origin = { 0, 0, layerIndexInTexture };
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
    std::cout << "Buffer " << buffer << ": Image uploaded to GPU." << std::endl;

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
            UniformsData uniformsData = { static_cast<int32_t>(layerIndexInTexture), {0,0,0} };
            queue_.WriteBuffer(uniformBuffers_[frontBuffer], 0, &uniformsData, sizeof(UniformsData));
        }
    }
}

void ImageFlasher::render(wgpu::RenderPassEncoder& pass) {
    int frontBuffer = bufferIndex_;

    if (imagesInBuffer_[frontBuffer] == 0) {
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
        UniformsData uniformsData = { static_cast<int32_t>(layerIndexInTexture), {0,0,0} };
        queue_.WriteBuffer(uniformBuffers_[bufferIndex_], 0, &uniformsData, sizeof(UniformsData));
    }
}

// Helper function to create shader modules with error checking
wgpu::ShaderModule createShaderModule(wgpu::Device device, const char* code, const char* shaderName) {
    wgpu::ShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.code = code;

    wgpu::ShaderModuleDescriptor shaderDesc = {};
    shaderDesc.nextInChain = &wgslDesc;

    wgpu::ShaderModule module = device.CreateShaderModule(&shaderDesc);

    if (!module) {
        std::cerr << "Failed to create shader module: " << shaderName << std::endl;
    } else {
        std::cout << "Shader module created successfully: " << shaderName << std::endl;
    }

    return module;
}

ImageFlasher* imageFlasher = nullptr;

// Function to create the render pipeline
void createRenderPipeline() {
    std::cout << "Creating render pipeline." << std::endl;
    wgpu::ShaderModule vsModule = createShaderModule(device, vertexShaderCode, "Vertex Shader");
    wgpu::ShaderModule fsModule = createShaderModule(device, fragmentShaderCode, "Fragment Shader");

    if (!vsModule || !fsModule) {
        std::cerr << "Failed to create shader modules. Aborting pipeline creation." << std::endl;
        return;
    }

    wgpu::PipelineLayout pipelineLayout = imageFlasher->getPipelineLayout();

    wgpu::RenderPipelineDescriptor desc = {};

    desc.vertex.module = vsModule;
    desc.vertex.entryPoint = "main";
    desc.vertex.bufferCount = 0;
    desc.vertex.buffers = nullptr;

    wgpu::ColorTargetState colorTarget = {};
    colorTarget.format = swapChainFormat;

    wgpu::FragmentState fragmentState = {};
    fragmentState.module = fsModule;
    fragmentState.entryPoint = "main";
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTarget;

    desc.fragment = &fragmentState;
    desc.layout = pipelineLayout;
    desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    desc.primitive.stripIndexFormat = wgpu::IndexFormat::Undefined;
    desc.primitive.frontFace = wgpu::FrontFace::CCW;
    desc.primitive.cullMode = wgpu::CullMode::None;

    desc.multisample.count = 1;
    desc.multisample.mask = ~0u;
    desc.multisample.alphaToCoverageEnabled = false;

    pipeline = device.CreateRenderPipeline(&desc);
    if (!pipeline) {
        std::cerr << "Failed to create render pipeline." << std::endl;
    } else {
        std::cout << "Render pipeline created successfully." << std::endl;
    }
}

// New class: ImageLoader
// This class scans a given folder for PNG images. It keeps track of which images it has loaded,
// so it won't load them more than once. If it finds new images, it loads them, resizes them to 512x512,
// and pushes them into the ImageFlasher via pushImageCallback. It also maintains a FIFO buffer of loaded filenames
// so that it doesn't grow unbounded. If the folder changes, newly added images will appear.

class ImageLoader {
public:
    ImageLoader(const std::string& folderPath,
                std::function<void(const ImageData&)> pushImageCallback,
                size_t maxLoadedImages);
    ~ImageLoader();

private:
    void run();
    bool loadAndResizeImage(const std::string& filePath, ImageData& imageOut);

    std::string folderPath_;
    std::function<void(const ImageData&)> pushImageCallback_;
    size_t maxLoadedImages_;

    std::atomic<bool> running_;
    std::thread thread_;

    // To track which images have been loaded
    // We'll use a deque for FIFO and a set for quick lookup
    std::deque<std::string> loadedImagesQueue_;
    std::set<std::string> loadedImagesSet_;
    std::mutex loadedImagesMutex_;
};

ImageLoader::ImageLoader(const std::string& folderPath,
                         std::function<void(const ImageData&)> pushImageCallback,
                         size_t maxLoadedImages)
        : folderPath_(folderPath),
          pushImageCallback_(pushImageCallback),
          maxLoadedImages_(maxLoadedImages),
          running_(true) {
    thread_ = std::thread(&ImageLoader::run, this);
}

ImageLoader::~ImageLoader() {
    running_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
}

void ImageLoader::run() {
    std::cout << "ImageLoader thread started, monitoring folder: " << folderPath_ << std::endl;

    while (running_) {
        try {
            // Scan folder for PNG files
            for (auto& entry : std::filesystem::directory_iterator(folderPath_)) {
                if (!running_) break;
                if (entry.is_regular_file()) {
                    auto path = entry.path();
                    if (path.extension() == ".png") {
                        std::string filename = path.string();

                        // Check if we've already loaded this image
                        {
                            std::lock_guard<std::mutex> lock(loadedImagesMutex_);
                            if (loadedImagesSet_.find(filename) != loadedImagesSet_.end()) {
                                // Already loaded
                                continue;
                            }
                        }

                        // Load and resize image
                        ImageData imgData;
                        if (loadAndResizeImage(filename, imgData)) {
                            // Push image to ImageFlasher
                            pushImageCallback_(imgData);

                            // Add to loadedImagesSet_ and loadedImagesQueue_ (FIFO)
                            {
                                std::lock_guard<std::mutex> lock(loadedImagesMutex_);
                                loadedImagesQueue_.push_back(filename);
                                loadedImagesSet_.insert(filename);

                                // If we exceed maxLoadedImages_, pop oldest
                                while (loadedImagesQueue_.size() > maxLoadedImages_) {
                                    std::string oldest = loadedImagesQueue_.front();
                                    loadedImagesQueue_.pop_front();
                                    loadedImagesSet_.erase(oldest);
                                }
                            }
                        }
                    }
                }
            }
        } catch (std::exception& e) {
            std::cerr << "Exception scanning folder: " << e.what() << std::endl;
        }

        // Sleep for a short duration before rescanning
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "ImageLoader thread exiting." << std::endl;
}

bool ImageLoader::loadAndResizeImage(const std::string& filePath, ImageData& imageOut) {
    // Load image using stb_image
    int x, y, n;
    unsigned char* data = stbi_load(filePath.c_str(), &x, &y, &n, 4); // force RGBA
    if (!data) {
        std::cerr << "Failed to load image: " << filePath << std::endl;
        return false;
    }

    // We must resize to 512x512
    const int desiredWidth = 512;
    const int desiredHeight = 512;
    std::vector<unsigned char> resized(desiredWidth * desiredHeight * 4);

    int result = stbir_resize_uint8(data, x, y, 0,
                                    resized.data(), desiredWidth, desiredHeight, 0,
                                    4); // channels = 4 (RGBA)
    stbi_image_free(data);
    if (!result) {
        std::cerr << "Failed to resize image: " << filePath << std::endl;
        return false;
    }

    imageOut.width = desiredWidth;
    imageOut.height = desiredHeight;
    imageOut.pixels = std::move(resized);

    std::cout << "Loaded and resized image: " << filePath << std::endl;
    return true;
}


std::unique_ptr<ImageLoader> imageLoader;

// We remove procedural generation and use ImageLoader now. The rest remains the same.

void initializeSwapChainAndPipeline(wgpu::Surface surface) {
    std::cout << "Initializing swap chain and pipeline." << std::endl;

    wgpu::SwapChainDescriptor swapChainDesc = {};
    swapChainDesc.format = wgpu::TextureFormat::BGRA8Unorm;
    swapChainDesc.usage = wgpu::TextureUsage::RenderAttachment;
    swapChainDesc.presentMode = wgpu::PresentMode::Fifo;

    swapChainFormat = swapChainDesc.format;

    double canvasWidth, canvasHeight;
    emscripten_get_element_css_size("canvas", &canvasWidth, &canvasHeight);
    std::cout << "Canvas size: " << canvasWidth << "x" << canvasHeight << std::endl;

    swapChainDesc.width = static_cast<uint32_t>(canvasWidth);
    swapChainDesc.height = static_cast<uint32_t>(canvasHeight);

    if (swapChainDesc.width == 0 || swapChainDesc.height == 0) {
        std::cerr << "Invalid canvas size." << std::endl;
        return;
    }

    swapChain = device.CreateSwapChain(surface, &swapChainDesc);
    std::cout << "Swap chain created." << std::endl;

    if (!swapChain) {
        std::cerr << "Failed to create swap chain." << std::endl;
        return;
    }

    constexpr uint32_t RING_BUFFER_SIZE = 1024;
    constexpr float IMAGE_SWITCH_INTERVAL = 1.0f / 60.0f; // ~60 times per second
    imageFlasher = new ImageFlasher(device, RING_BUFFER_SIZE, IMAGE_SWITCH_INTERVAL);
    std::cout << "ImageFlasher instance created." << std::endl;

    // Instead of imageGenThread, we now create our ImageLoader
    // Let's assume the folder path is "./images"
    // We'll monitor this folder for new PNG images
    // We'll have a maxLoadedImages to limit how many unique images we remember having loaded
    size_t YOUR_MAX_LOADED_IMAGES = 100; // for example
    imageLoader = std::make_unique<ImageLoader>("./images",
                                                [](const ImageData& img){ imageFlasher->pushImage(img); },
                                                YOUR_MAX_LOADED_IMAGES);

    createRenderPipeline();
    std::cout << "Render pipeline created." << std::endl;

    // Start the main loop
    emscripten_request_animation_frame_loop(frame, nullptr);
}

// Callback when device request ends
void onDeviceRequestEnded(WGPURequestDeviceStatus status,
                          WGPUDevice cDevice,
                          const char* message,
                          void* userdata) {
    if (status == WGPURequestDeviceStatus_Success) {
        device = wgpu::Device::Acquire(cDevice);
        queue = device.GetQueue();
        std::cout << "WebGPU device acquired." << std::endl;

        device.SetUncapturedErrorCallback(HandleUncapturedError, nullptr);

        WGPUSurface surface = static_cast<WGPUSurface>(userdata);
        initializeSwapChainAndPipeline(wgpu::Surface::Acquire(surface));
    } else {
        std::cerr << "Failed to create device: " << (message ? message : "Unknown error") << std::endl;
    }
}

// Callback when adapter request ends
void onAdapterRequestEnded(WGPURequestAdapterStatus status,
                           WGPUAdapter cAdapter,
                           const char* message,
                           void* userdata) {
    if (status == WGPURequestAdapterStatus_Success) {
        wgpu::Adapter adapter = wgpu::Adapter::Acquire(cAdapter);
        std::cout << "WebGPU adapter acquired." << std::endl;

        wgpu::DeviceDescriptor deviceDesc = {};
        deviceDesc.label = "My Device";

        adapter.RequestDevice(&deviceDesc, onDeviceRequestEnded, userdata);
    } else {
        std::cerr << "Failed to get WebGPU adapter: " << (message ? message : "Unknown error") << std::endl;
    }
}

// Frame callback
EM_BOOL frame(double time, void* userData) {
    if (!swapChain) {
        std::cerr << "Swap chain not initialized." << std::endl;
        return EM_FALSE;
    }

    wgpu::TextureView backbuffer = swapChain.GetCurrentTextureView();
    if (!backbuffer) {
        std::cerr << "Failed to get current texture view." << std::endl;
        return EM_FALSE;
    }

    imageFlasher->update();

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();

    wgpu::RenderPassColorAttachment colorAttachment = {};
    colorAttachment.view = backbuffer;
    colorAttachment.loadOp = wgpu::LoadOp::Clear;
    colorAttachment.storeOp = wgpu::StoreOp::Store;
    colorAttachment.clearValue = { 0.3f, 0.3f, 0.3f, 1.0f };

    wgpu::RenderPassDescriptor renderPassDesc = {};
    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &colorAttachment;

    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPassDesc);
    pass.SetPipeline(pipeline);
    imageFlasher->render(pass);
    pass.Draw(6, 1, 0, 0);
    pass.End();

    wgpu::CommandBuffer cmdBuffer = encoder.Finish();
    queue.Submit(1, &cmdBuffer);

    // Do not call swapChain.Present(); it's handled implicitly by the browser with requestAnimationFrame
    return EM_TRUE;
}

// Cleanup function
void cleanup() {
    std::cout << "Cleaning up resources." << std::endl;
    // imageGenRunning = false; // Not used now since we replaced image generation with ImageLoader
    // if (imageGenThread.joinable()) {
    //     imageGenThread.join();
    //     std::cout << "Image generation thread joined." << std::endl;
    // }
    imageLoader.reset();
    delete imageFlasher;
    imageFlasher = nullptr;
    std::cout << "ImageFlasher deleted." << std::endl;
}

// Entry point
int main() {
    std::cout << "Starting application." << std::endl;

    WGPUInstanceDescriptor instanceDesc = {};
    WGPUInstance instance = wgpuCreateInstance(&instanceDesc);
    std::cout << "WebGPU instance created." << std::endl;

    WGPURequestAdapterOptions adapterOpts = {};
    adapterOpts.powerPreference = WGPUPowerPreference_HighPerformance;

    WGPUSurfaceDescriptorFromCanvasHTMLSelector canvDesc = {};
    canvDesc.chain.sType = WGPUSType_SurfaceDescriptorFromCanvasHTMLSelector;
    canvDesc.selector = "canvas";

    WGPUSurfaceDescriptor surfDesc = {};
    surfDesc.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&canvDesc);

    WGPUSurface surface = wgpuInstanceCreateSurface(instance, &surfDesc);
    if (!surface) {
        std::cerr << "Failed to create WebGPU surface." << std::endl;
        return -1;
    }

    surfaceGlobal = wgpu::Surface::Acquire(surface);
    std::cout << "WebGPU surface created." << std::endl;

    wgpuInstanceRequestAdapter(
            instance,
            &adapterOpts,
            onAdapterRequestEnded,
            surface
    );

    emscripten_exit_with_live_runtime();

    return 0;
}
