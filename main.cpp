#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>

#include <emscripten.h>
#include <emscripten/html5.h> // For emscripten_request_animation_frame_loop
#include <emscripten/html5_webgpu.h>

#include <webgpu/webgpu_cpp.h>

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
[[block]] struct Uniforms {
    layerIndex : u32;
};
[[group(0), binding(0)]] var<uniform> uniforms : Uniforms;
[[group(0), binding(1)]] var textureArray : texture_2d_array<f32>;
[[group(0), binding(2)]] var sampler0 : sampler;

@fragment
fn main(@location(0) fragUV : vec2<f32>) -> @location(0) vec4<f32> {
    let layer = uniforms.layerIndex;
    let color = textureSample(textureArray, sampler0, fragUV, i32(layer));
    return color;
}
)";

// Global variables for device and so on
wgpu::Device device;
wgpu::Queue queue;
wgpu::SwapChain swapChain;
wgpu::RenderPipeline pipeline;
wgpu::Surface surfaceGlobal;
wgpu::TextureFormat swapChainFormat;

class ImageFlasher;

// Forward declaration
EM_BOOL frame(double time, void* userData);

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

class ImageFlasher {
public:
    ImageFlasher(wgpu::Device device, uint32_t ringBufferSize);
    ~ImageFlasher();

    void pushImage(const ImageData& image);
    void update(); // Call in main loop
    void render(wgpu::RenderPassEncoder& pass);

    wgpu::PipelineLayout getPipelineLayout() const { return pipelineLayout_; }
    wgpu::BindGroup getBindGroup() const { return bindGroup_; }

private:
    void uploadImage(const ImageData& image);

    wgpu::Device device_;
    wgpu::Queue queue_;

    wgpu::Texture textureArray_;
    wgpu::Sampler sampler_;
    wgpu::TextureView textureView_;
    wgpu::BindGroupLayout bindGroupLayout_;
    wgpu::BindGroup bindGroup_;
    wgpu::Buffer uniformBuffer_;
    wgpu::PipelineLayout pipelineLayout_;

    uint32_t ringBufferSize_;
    uint32_t textureWidth_ = 512;
    uint32_t textureHeight_ = 512;
    uint32_t writeIndex_;
    uint32_t displayIndex_;

    mutable std::mutex mutex_;
    ThreadSafeQueue<ImageData> imageQueue_;
};

ImageFlasher::ImageFlasher(wgpu::Device device, uint32_t ringBufferSize)
        : device_(device),
          queue_(device.GetQueue()),
          ringBufferSize_(ringBufferSize),
          writeIndex_(0),
          displayIndex_(0) {
    // Create texture array
    wgpu::TextureDescriptor textureDesc = {};
    textureDesc.size.width = textureWidth_;
    textureDesc.size.height = textureHeight_;
    textureDesc.size.depthOrArrayLayers = ringBufferSize_;
    textureDesc.format = wgpu::TextureFormat::RGBA8Unorm;
    textureDesc.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
    textureArray_ = device_.CreateTexture(&textureDesc);

    // Create sampler
    wgpu::SamplerDescriptor samplerDesc = {};
    samplerDesc.addressModeU = wgpu::AddressMode::ClampToEdge;
    samplerDesc.addressModeV = wgpu::AddressMode::ClampToEdge;
    samplerDesc.magFilter = wgpu::FilterMode::Linear;
    samplerDesc.minFilter = wgpu::FilterMode::Linear;
    sampler_ = device_.CreateSampler(&samplerDesc);

    // Create uniform buffer for layer index
    wgpu::BufferDescriptor uniformBufferDesc = {};
    uniformBufferDesc.size = sizeof(uint32_t);
    uniformBufferDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniformBuffer_ = device_.CreateBuffer(&uniformBufferDesc);

    // Create bind group layout
    wgpu::BindGroupLayoutEntry bglEntries[3] = {};
    bglEntries[0].binding = 0;
    bglEntries[0].visibility = wgpu::ShaderStage::Fragment;
    bglEntries[0].buffer.type = wgpu::BufferBindingType::Uniform;
    bglEntries[0].buffer.minBindingSize = sizeof(uint32_t);

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

    // Create pipeline layout
    wgpu::PipelineLayoutDescriptor plDesc = {};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &bindGroupLayout_;
    pipelineLayout_ = device_.CreatePipelineLayout(&plDesc);

    // Create texture view
    wgpu::TextureViewDescriptor viewDesc = {};
    viewDesc.dimension = wgpu::TextureViewDimension::e2DArray;
    textureView_ = textureArray_.CreateView(&viewDesc);

    // Create bind group
    wgpu::BindGroupEntry bgEntries[3] = {};
    bgEntries[0].binding = 0;
    bgEntries[0].buffer = uniformBuffer_;
    bgEntries[0].offset = 0;
    bgEntries[0].size = sizeof(uint32_t);

    bgEntries[1].binding = 1;
    bgEntries[1].textureView = textureView_;

    bgEntries[2].binding = 2;
    bgEntries[2].sampler = sampler_;

    wgpu::BindGroupDescriptor bgDesc = {};
    bgDesc.layout = bindGroupLayout_;
    bgDesc.entryCount = 3;
    bgDesc.entries = bgEntries;

    bindGroup_ = device_.CreateBindGroup(&bgDesc);
}

ImageFlasher::~ImageFlasher() {
    // Destructor (empty)
}

void ImageFlasher::pushImage(const ImageData& image) {
    imageQueue_.push(image);
}

void ImageFlasher::uploadImage(const ImageData& image) {
    // Upload the image to the texture array at the current write index
    wgpu::ImageCopyTexture dst = {};
    dst.texture = textureArray_;
    dst.mipLevel = 0;
    dst.origin = { 0, 0, writeIndex_ };
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
}

void ImageFlasher::update() {
    // Process any new images
    while (true) {
        ImageData image;
        if (!imageQueue_.tryPop(image)) {
            break;
        }

        // Upload image to the texture array
        uploadImage(image);

        // Update write index
        writeIndex_ = (writeIndex_ + 1) % ringBufferSize_;
    }

    // Update display index
    displayIndex_ = (displayIndex_ + 1) % ringBufferSize_;

    // Update uniform buffer with display index
    queue_.WriteBuffer(uniformBuffer_, 0, &displayIndex_, sizeof(uint32_t));
}

void ImageFlasher::render(wgpu::RenderPassEncoder& pass) {
    // Set the bind group
    pass.SetBindGroup(0, bindGroup_);
}

// Helper function to create shader modules
wgpu::ShaderModule createShaderModule(const char* code) {
    wgpu::ShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.code = code;

    wgpu::ShaderModuleDescriptor shaderDesc = {};
    shaderDesc.nextInChain = &wgslDesc;

    return device.CreateShaderModule(&shaderDesc);
}

ImageFlasher* imageFlasher = nullptr;

// Function to create the render pipeline
void createRenderPipeline() {
    wgpu::ShaderModule vsModule = createShaderModule(vertexShaderCode);
    wgpu::ShaderModule fsModule = createShaderModule(fragmentShaderCode);

    // Use the bind group layout from ImageFlasher
    wgpu::PipelineLayout pipelineLayout = imageFlasher->getPipelineLayout();

    wgpu::RenderPipelineDescriptor desc = {};

    // Vertex state
    desc.vertex.module = vsModule;
    desc.vertex.entryPoint = "main";
    desc.vertex.bufferCount = 0;
    desc.vertex.buffers = nullptr;

    // Fragment state
    wgpu::ColorTargetState colorTarget = {};
    colorTarget.format = swapChainFormat; // Use swapChainFormat variable

    wgpu::FragmentState fragmentState = {};
    fragmentState.module = fsModule;
    fragmentState.entryPoint = "main";
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTarget;

    desc.fragment = &fragmentState;

    // Other states
    desc.layout = pipelineLayout; // Use the pipeline layout from ImageFlasher

    desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    desc.primitive.stripIndexFormat = wgpu::IndexFormat::Undefined;
    desc.primitive.frontFace = wgpu::FrontFace::CCW;
    desc.primitive.cullMode = wgpu::CullMode::None;

    desc.multisample.count = 1;
    desc.multisample.mask = ~0u;
    desc.multisample.alphaToCoverageEnabled = false;

    pipeline = device.CreateRenderPipeline(&desc);
}

// Function to initialize the swap chain and pipeline
void initializeSwapChainAndPipeline(wgpu::Surface surface) {
    // Create swap chain
    wgpu::SwapChainDescriptor swapChainDesc = {};
    swapChainDesc.format = wgpu::TextureFormat::BGRA8Unorm; // Ensure this matches pipeline
    swapChainDesc.usage = wgpu::TextureUsage::RenderAttachment;
    swapChainDesc.presentMode = wgpu::PresentMode::Fifo;

    swapChainFormat = swapChainDesc.format;

    // Get canvas size
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

    if (!swapChain) {
        std::cerr << "Failed to create swap chain." << std::endl;
        return;
    }

    // Create ImageFlasher instance
    imageFlasher = new ImageFlasher(device, 1024);

    // Create pipeline
    createRenderPipeline();

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

        // Now that we have the device, initialize swap chain and pipeline
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

        // Request device
        wgpu::DeviceDescriptor deviceDesc = {};
        deviceDesc.label = "My Device";

        adapter.RequestDevice(&deviceDesc, onDeviceRequestEnded, userdata);
    } else {
        std::cerr << "Failed to get WebGPU adapter: " << (message ? message : "Unknown error") << std::endl;
    }
}

// Image generation thread
std::atomic<bool> imageGenRunning = true;
std::thread imageGenThread;

void imageGenerationFunction() {
    while (imageGenRunning) {
        ImageData image;
        image.width = 512;
        image.height = 512;
        image.pixels.resize(image.width * image.height * 4);

        // Fill with random data
        for (size_t i = 0; i < image.pixels.size(); ++i) {
            image.pixels[i] = rand() % 256;
        }

        // Push the image to ImageFlasher
        std::cout<<"pushing"<<std::endl;
        imageFlasher->pushImage(image);

        // Sleep for a while
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Main rendering loop
EM_BOOL frame(double time, void* userData) {
    // Ensure swap chain is valid
    if (!swapChain) {
        std::cerr << "Swap chain not initialized." << std::endl;
        return EM_FALSE;
    }

    wgpu::TextureView backbuffer = swapChain.GetCurrentTextureView();
    if (!backbuffer) {
        std::cerr << "Failed to get current texture view." << std::endl;
        return EM_FALSE;
    }

    // Call imageFlasher->update()
    imageFlasher->update();

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();

    wgpu::RenderPassColorAttachment colorAttachment = {};
    colorAttachment.view = backbuffer;
    colorAttachment.loadOp = wgpu::LoadOp::Clear;
    colorAttachment.storeOp = wgpu::StoreOp::Store;
    colorAttachment.clearValue = { 0.3f, 0.3f, 0.3f, 1.0f }; // Gray background

    wgpu::RenderPassDescriptor renderPassDesc = {};
    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &colorAttachment;

    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPassDesc);

    pass.SetPipeline(pipeline);

    // Set the bind group
    pass.SetBindGroup(0, imageFlasher->getBindGroup());

    pass.Draw(6, 1, 0, 0);
    pass.End();

    wgpu::CommandBuffer cmdBuffer = encoder.Finish();
    queue.Submit(1, &cmdBuffer);

    // Continue the loop
    return EM_TRUE;
}


// Entry point
int main() {
    // Create a WGPUInstance
    WGPUInstanceDescriptor instanceDesc = {};
    WGPUInstance instance = wgpuCreateInstance(&instanceDesc);

    // Get the default WebGPU adapter
    WGPURequestAdapterOptions adapterOpts = {};
    adapterOpts.powerPreference = WGPUPowerPreference_HighPerformance;

    // Create surface from canvas
    WGPUSurfaceDescriptorFromCanvasHTMLSelector canvDesc = {};
    canvDesc.chain.sType = WGPUSType_SurfaceDescriptorFromCanvasHTMLSelector;
    canvDesc.selector = "canvas"; // Ensure your HTML has a <canvas id="canvas"></canvas>

    WGPUSurfaceDescriptor surfDesc = {};
    surfDesc.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&canvDesc);

    WGPUSurface surface = wgpuInstanceCreateSurface(instance, &surfDesc);
    if (!surface) {
        std::cerr << "Failed to create WebGPU surface." << std::endl;
        return -1;
    }

    surfaceGlobal = wgpu::Surface::Acquire(surface);;

    // Request adapter
    wgpuInstanceRequestAdapter(
            instance,
            &adapterOpts,
            onAdapterRequestEnded,
            surface // Pass the surface as userdata
    );

    // Start image generation thread
    imageGenRunning = true;
    imageGenThread = std::thread(imageGenerationFunction);

    // Keep the runtime alive without blocking the main thread
    emscripten_exit_with_live_runtime();

    // Clean up (This code is unreachable in Emscripten unless you call emscripten_cancel_main_loop)
    imageGenRunning = false;
    if (imageGenThread.joinable()) {
        imageGenThread.join();
    }
    delete imageFlasher;

    return 0;
}