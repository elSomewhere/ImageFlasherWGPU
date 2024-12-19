#include <iostream>
#include <vector>

#include <emscripten.h>
#include <emscripten/html5.h> // For emscripten_request_animation_frame_loop
#include <emscripten/html5_webgpu.h>

#include <webgpu/webgpu_cpp.h>

// Shader code remains the same...
const char* vertexShaderCode = R"(
@vertex
fn main(@builtin(vertex_index) VertexIndex: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>(0.5, -0.5),
        vec2<f32>(0.5, 0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>(0.5, 0.5),
        vec2<f32>(-0.5, 0.5)
    );
    return vec4<f32>(pos[VertexIndex], 0.0, 1.0);
}
)";

const char* fragmentShaderCode = R"(
@fragment
fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.5, 0.0, 1.0); // Orange color
}
)";

// Global variables for device and so on
wgpu::Device device;
wgpu::Queue queue;
wgpu::SwapChain swapChain;
wgpu::RenderPipeline pipeline;

// Forward declaration
EM_BOOL frame(double time, void* userData);

// Helper function to create shader modules
wgpu::ShaderModule createShaderModule(const char* code) {
    wgpu::ShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.code = code;

    wgpu::ShaderModuleDescriptor shaderDesc = {};
    shaderDesc.nextInChain = &wgslDesc;

    return device.CreateShaderModule(&shaderDesc);
}

// Function to create the render pipeline
void createRenderPipeline() {
    wgpu::ShaderModule vsModule = createShaderModule(vertexShaderCode);
    wgpu::ShaderModule fsModule = createShaderModule(fragmentShaderCode);

    // Create pipeline layout
    wgpu::PipelineLayoutDescriptor layoutDesc = {};
    layoutDesc.bindGroupLayoutCount = 0;
    layoutDesc.bindGroupLayouts = nullptr;

    wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&layoutDesc);

    wgpu::RenderPipelineDescriptor desc = {};

    // Vertex state
    desc.vertex.module = vsModule;
    desc.vertex.entryPoint = "main";
    desc.vertex.bufferCount = 0;
    desc.vertex.buffers = nullptr;

    // Fragment state
    wgpu::ColorTargetState colorTarget = {};
    colorTarget.format = wgpu::TextureFormat::BGRA8Unorm; // Ensure this matches swap chain

    wgpu::FragmentState fragmentState = {};
    fragmentState.module = fsModule;
    fragmentState.entryPoint = "main";
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTarget;

    desc.fragment = &fragmentState;

    // Other states
    desc.layout = pipelineLayout; // Use the created layout

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
    pass.Draw(6, 1, 0, 0);
    pass.End();

    wgpu::CommandBuffer cmdBuffer = encoder.Finish();
    queue.Submit(1, &cmdBuffer);

    // Return EM_TRUE to keep the loop running
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
    canvDesc.selector = "canvas"; // Removed '#' prefix

    WGPUSurfaceDescriptor surfDesc = {};
    surfDesc.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&canvDesc);

    WGPUSurface surface = wgpuInstanceCreateSurface(instance, &surfDesc);
    if (!surface) {
        std::cerr << "Failed to create WebGPU surface." << std::endl;
        return -1;
    }

    // Request adapter
    wgpuInstanceRequestAdapter(
            instance,
            &adapterOpts,
            onAdapterRequestEnded,
            surface // Pass the surface as userdata
    );

    // Run the Emscripten main loop
    emscripten_set_main_loop([](){}, 0, 0);

    return 0;
}
