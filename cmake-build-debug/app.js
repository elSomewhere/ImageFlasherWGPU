// app.js

// We'll rely on Module['onRuntimeInitialized'] from the Emscripten module.
// That means once WASM is ready, we set up our WebSocket and attach UI events.

Module['onRuntimeInitialized'] = () => {
    console.log("WASM runtime initialized. Setting up WebSocket + UI...");

    // ------------------------------------------------------------------------
    // 1) Connect to the Python WebSocket server that streams random images.
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
    // 2) Hook up UI Sliders & Button to the C++ side
    //    (We can use Module.ccall or cwrap as you prefer.)
    // ------------------------------------------------------------------------

    // cwrap references for all exported C++ functions we need:
    const setFadeFactor          = Module.cwrap('setFadeFactor', null, ['number']);
    const setImageSwitchInterval = Module.cwrap('setImageSwitchInterval', null, ['number']);
    const getBufferUsage         = Module.cwrap('getBufferUsage', 'number', []);
    const getRingBufferSize      = Module.cwrap('getRingBufferSize', 'number', []);
    const setMaxUploadsPerFrame  = Module.cwrap('setMaxUploadsPerFrame', null, ['number']);
    const setScrollingSpeed      = Module.cwrap('setScrollingSpeed', null, ['number', 'number']);
    const setScrollingOffset     = Module.cwrap('setScrollingOffset', null, ['number', 'number']);
    const setTileFactor          = Module.cwrap('setTileFactor', null, ['number']);

    // Grab all relevant DOM elements:
    const fadeSlider       = document.getElementById('fadeSlider');
    const fadeValue        = document.getElementById('fadeValue');
    const switchSlider     = document.getElementById('switchSlider');
    const switchValue      = document.getElementById('switchValue');
    const tileSlider       = document.getElementById('tileSlider');
    const tileValue        = document.getElementById('tileValue');
    const scrollSpeedX     = document.getElementById('scrollSpeedX');
    const scrollSpeedXVal  = document.getElementById('scrollSpeedXVal');
    const scrollSpeedY     = document.getElementById('scrollSpeedY');
    const scrollSpeedYVal  = document.getElementById('scrollSpeedYVal');
    const scrollOffsetX    = document.getElementById('scrollOffsetX');
    const scrollOffsetXVal = document.getElementById('scrollOffsetXVal');
    const scrollOffsetY    = document.getElementById('scrollOffsetY');
    const scrollOffsetYVal = document.getElementById('scrollOffsetYVal');
    const uploadsSlider    = document.getElementById('uploadsSlider');
    const uploadsValue     = document.getElementById('uploadsValue');
    const updateBufferUsageBtn = document.getElementById('updateBufferUsage');
    const bufferUsageLabel = document.getElementById('bufferUsageLabel');

    // Helper: update fade factor
    fadeSlider.addEventListener('input', () => {
        const val = parseFloat(fadeSlider.value);
        fadeValue.textContent = val.toFixed(2);
        setFadeFactor(val);
    });

    // Helper: update switch interval
    switchSlider.addEventListener('input', () => {
        const val = parseFloat(switchSlider.value);
        switchValue.textContent = val.toFixed(2);
        setImageSwitchInterval(val);
    });

    // Helper: update tile factor
    tileSlider.addEventListener('input', () => {
        const val = parseInt(tileSlider.value, 10);
        tileValue.textContent = val;
        setTileFactor(val);
    });

    // Helper: update scrolling speed X
    scrollSpeedX.addEventListener('input', () => {
        const val = parseFloat(scrollSpeedX.value);
        scrollSpeedXVal.textContent = val.toFixed(2);
        const speedY = parseFloat(scrollSpeedY.value);
        setScrollingSpeed(val, speedY);
    });

    // Helper: update scrolling speed Y
    scrollSpeedY.addEventListener('input', () => {
        const val = parseFloat(scrollSpeedY.value);
        scrollSpeedYVal.textContent = val.toFixed(2);
        const speedX = parseFloat(scrollSpeedX.value);
        setScrollingSpeed(speedX, val);
    });

    // Helper: update scrolling offset X
    scrollOffsetX.addEventListener('input', () => {
        const val = parseFloat(scrollOffsetX.value);
        scrollOffsetXVal.textContent = val.toFixed(2);
        const oy = parseFloat(scrollOffsetY.value);
        setScrollingOffset(val, oy);
    });

    // Helper: update scrolling offset Y
    scrollOffsetY.addEventListener('input', () => {
        const val = parseFloat(scrollOffsetY.value);
        scrollOffsetYVal.textContent = val.toFixed(2);
        const ox = parseFloat(scrollOffsetX.value);
        setScrollingOffset(ox, val);
    });

    // Helper: update max uploads
    uploadsSlider.addEventListener('input', () => {
        const val = parseInt(uploadsSlider.value, 10);
        uploadsValue.textContent = val;
        setMaxUploadsPerFrame(val);
    });

    // Button: check buffer usage
    updateBufferUsageBtn.addEventListener('click', () => {
        const usage = getBufferUsage();
        const capacity = getRingBufferSize();
        bufferUsageLabel.textContent = `Buffer usage: ${usage} / ${capacity}`;
    });

    // ------------------------------------------------------------------------
    // 3) Initialize default slider values, so our C++ code starts with something
    // ------------------------------------------------------------------------
    fadeSlider.dispatchEvent(new Event('input'));
    switchSlider.dispatchEvent(new Event('input'));
    tileSlider.dispatchEvent(new Event('input'));
    scrollSpeedX.dispatchEvent(new Event('input'));
    scrollSpeedY.dispatchEvent(new Event('input'));
    scrollOffsetX.dispatchEvent(new Event('input'));
    scrollOffsetY.dispatchEvent(new Event('input'));
    uploadsSlider.dispatchEvent(new Event('input'));

    console.log("All UI events + WebSocket are set up!");
};
