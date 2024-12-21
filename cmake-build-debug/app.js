// index.js

// Assume Module is the Emscripten module object.
// Emscripten injects a `Module` object to the global scope if using the default shell.
// Ensure that the wasm code has been loaded before running this. If needed, use Module.onRuntimeInitialized.
Module['onRuntimeInitialized'] = () => {
    console.log("Runtime initialized, setting up WebSocket...");
    console.log("Attempting to connect to WebSocket on ws://localhost:5000");
    let ws = new WebSocket("ws://127.0.0.1:5010");
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
        console.log("WebSocket connected");
    };
    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
    };
    ws.onmessage = (event) => {
        console.log("Received data from WS");
        let arrayBuffer = event.data; // ArrayBuffer
        let byteArray = new Uint8Array(arrayBuffer);

        // Allocate memory in WASM heap
        let ptr = Module._malloc(byteArray.length);
        Module.HEAPU8.set(byteArray, ptr);

        // Call the C++ function
        // onImageReceived(uint8_t* data, int length)
        Module.ccall('onImageReceived', null, ['number', 'number'], [ptr, byteArray.length]);

        Module._free(ptr);
    };
    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
    };
};