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

#include <emscripten.h>
#include <emscripten/html5.h> // For emscripten_request_animation_frame_loop
#include <emscripten/html5_webgpu.h>

#include <webgpu/webgpu_cpp.h>

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

    // Initialize buffer indices and counters
    for (int b = 0; b < 2; ++b) {
        writeIndex_[b] = 0;
        displayIndex_[b] = 0;
        imagesInBuffer_[b] = 0;
        lastSwitchTime_[b] = std::chrono::steady_clock::now();
    }

    // Compute number of texture arrays needed
    uint32_t numTextureArrays = (ringBufferSize_ + maxLayersPerArray - 1) / maxLayersPerArray;
    std::cout << "Number of texture arrays: " << numTextureArrays << std::endl;

    for (int b = 0; b < 2; ++b) {
        textureArrays_[b].resize(numTextureArrays);
        textureViews_[b].resize(numTextureArrays);
        bindGroups_[b].resize(numTextureArrays);
    }

    // Create sampler
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

// Upload image to the texture array for a specific buffer (FIFO logic)
void ImageFlasher::uploadImage(const ImageData& image, int buffer) {
    // If ring buffer full, discard oldest by incrementing displayIndex_
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

    // Update writeIndex_
    writeIndex_[buffer] = (writeIndex_[buffer] + 1) % ringBufferSize_;
}

// Update method:
// - Process new images (upload to back buffer)
// - If images were uploaded, swap buffers
// - Advance display index of front buffer based on imageSwitchInterval_
void ImageFlasher::update() {
    int frontBuffer = bufferIndex_;
    int backBuffer = 1 - frontBuffer;

    bool uploadedAnyImage = false;
    // Process all new images for this frame
    while (true) {
        ImageData image;
        if (!imageQueue_.tryPop(image)) {
            break;
        }
        uploadImage(image, backBuffer);
        uploadedAnyImage = true;
    }

    // If images were uploaded, swap buffers so new images become visible next time
    if (uploadedAnyImage) {
        swapBuffers();
        frontBuffer = bufferIndex_;
    }

    // Handle time-based switching on the front buffer
    // Only advance if we have at least one image
    if (imagesInBuffer_[frontBuffer] > 0) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed = now - lastSwitchTime_[frontBuffer];
        if (elapsed.count() >= imageSwitchInterval_) {
            // Advance display index
            displayIndex_[frontBuffer] = (displayIndex_[frontBuffer] + 1) % imagesInBuffer_[frontBuffer];
            lastSwitchTime_[frontBuffer] = now;

            // Update uniform buffer
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
        // No images, just bind the first bind group to avoid errors (blank)
        pass.SetBindGroup(0, bindGroups_[frontBuffer][0]);
        return;
    }

    uint32_t globalLayerIndex = displayIndex_[frontBuffer] % ringBufferSize_;
    uint32_t textureArrayIndex = globalLayerIndex / maxLayersPerArray;
    pass.SetBindGroup(0, bindGroups_[frontBuffer][textureArrayIndex]);
}

void ImageFlasher::swapBuffers() {
    bufferIndex_ = 1 - bufferIndex_;
    // Reset the lastSwitchTime_ for the new front buffer to ensure timing remains consistent.
    lastSwitchTime_[bufferIndex_] = std::chrono::steady_clock::now();

    // Also update the uniform buffer for the newly front buffer to ensure correct image is displayed immediately
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

    // Create ImageFlasher instance with double buffering and interval
    constexpr uint32_t RING_BUFFER_SIZE = 1024;
    constexpr float IMAGE_SWITCH_INTERVAL = 1.0f / 60.0f; // switch images ~60 times per second
    imageFlasher = new ImageFlasher(device, RING_BUFFER_SIZE, IMAGE_SWITCH_INTERVAL);
    std::cout << "ImageFlasher instance created." << std::endl;

    // Start image generation thread
    // Start image generation thread
    imageGenRunning = true;
    imageGenThread = std::thread([]() {
        std::cout << "Image generation thread started." << std::endl;
        while (imageGenRunning) {
            ImageData image;
            image.width = 512;
            image.height = 512;
            image.pixels.resize(image.width * image.height * 4);

            // Create a gradient from blue to green horizontally
            for (uint32_t y = 0; y < image.height; ++y) {
                for (uint32_t x = 0; x < image.width; ++x) {
                    float t = float(x) / float(image.width - 1);
                    uint8_t r = 0;
                    uint8_t g = (uint8_t)(255 * t);
                    uint8_t b = (uint8_t)(255 * (1.0f - t));
                    uint8_t a = 255;

                    size_t idx = (y * image.width + x) * 4;
                    image.pixels[idx + 0] = r;
                    image.pixels[idx + 1] = g;
                    image.pixels[idx + 2] = b;
                    image.pixels[idx + 3] = a;
                }
            }

            // Draw a few random circles
            int numCircles = 5;
            for (int c = 0; c < numCircles; ++c) {
                int cx = rand() % image.width;
                int cy = rand() % image.height;
                int radius = (rand() % 50) + 10;
                uint8_t cr = rand() % 256;
                uint8_t cg = rand() % 256;
                uint8_t cb = rand() % 256;
                uint8_t ca = 255;

                for (int y = -radius; y <= radius; ++y) {
                    for (int x = -radius; x <= radius; ++x) {
                        int nx = cx + x;
                        int ny = cy + y;
                        if (nx >= 0 && nx < (int)image.width && ny >= 0 && ny < (int)image.height) {
                            if (x*x + y*y <= radius*radius) {
                                size_t idx = (ny * image.width + nx) * 4;
                                image.pixels[idx + 0] = cr;
                                image.pixels[idx + 1] = cg;
                                image.pixels[idx + 2] = cb;
                                image.pixels[idx + 3] = ca;
                            }
                        }
                    }
                }
            }

            imageFlasher->pushImage(image);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "Image generation thread exiting." << std::endl;
    });
    std::cout << "Image generation thread started." << std::endl;

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

    // Update the imageFlasher (this processes new images and advances displayIndex_ based on time)
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

//    swapChain.Present();

    return EM_TRUE;
}

// Cleanup function
void cleanup() {
    std::cout << "Cleaning up resources." << std::endl;
    imageGenRunning = false;
    if (imageGenThread.joinable()) {
        imageGenThread.join();
        std::cout << "Image generation thread joined." << std::endl;
    }
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
