// app.js

// We'll rely on Module['onRuntimeInitialized'] from the Emscripten module.
// That means once WASM is ready, we set up our WebSocket and attach UI events.

Module['onRuntimeInitialized'] = () => {
    console.log("WASM runtime initialized. Setting up WebSocket...");

    // ------------------------------------------------------------------------
    // 1) Connect to Python WebSocket server that streams random images.
    //    - Adjust the port/host if needed
    // ------------------------------------------------------------------------
    const ws = new WebSocket("ws://127.0.0.1:5010");
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        console.log("WebSocket connected to ws://127.0.0.1:5010");
    };
    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
    };
    ws.onmessage = (event) => {
        // event.data is an ArrayBuffer containing PNG bytes
        let byteArray = new Uint8Array(event.data);

        // Allocate memory in WASM heap
        let ptr = Module._malloc(byteArray.length);
        Module.HEAPU8.set(byteArray, ptr);

        // Forward to C++: onImageReceived(uint8_t* data, int length)
        Module.ccall('onImageReceived', null, ['number', 'number'], [ptr, byteArray.length]);

        // Free the temporary buffer
        Module._free(ptr);
    };

    // ------------------------------------------------------------------------
    // 2) UI Sliders & Button
    // ------------------------------------------------------------------------
    const fadeSlider = document.getElementById('fadeSlider');
    const fadeValue = document.getElementById('fadeValue');
    const switchSlider = document.getElementById('switchSlider');
    const switchValue = document.getElementById('switchValue');
    const updateBufferUsageBtn = document.getElementById('updateBufferUsage');
    const bufferUsageLabel = document.getElementById('bufferUsageLabel');

    // When user drags the fade slider:
    fadeSlider.addEventListener('input', () => {
        const val = parseFloat(fadeSlider.value);
        fadeValue.textContent = val.toFixed(2);

        // Calls the C++ function: setFadeFactor(float factor)
        Module.ccall('setFadeFactor', null, ['number'], [val]);
    });

    // When user drags the switch interval slider:
    switchSlider.addEventListener('input', () => {
        const val = parseFloat(switchSlider.value);
        switchValue.textContent = val.toFixed(2);

        // Calls the C++ function: setImageSwitchInterval(float interval)
        Module.ccall('setImageSwitchInterval', null, ['number'], [val]);
    });

    // Button to query buffer usage from C++:
    updateBufferUsageBtn.addEventListener('click', () => {
        const usage = Module.ccall('getBufferUsage', 'number', [], []);
        const capacity = Module.ccall('getRingBufferSize', 'number', [], []);
        bufferUsageLabel.textContent = `Buffer usage: ${usage} / ${capacity}`;
    });
};
