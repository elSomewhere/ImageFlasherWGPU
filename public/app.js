// Enhanced Ikeda-inspired control system for ImageFlasherWGPU
// Extended with 8 new visual modes and advanced parameter controls

Module['onRuntimeInitialized'] = () => {
    console.log("WASM runtime initialized. Setting up extended Ikeda control system...");

    // ------------------------------------------------------------------------
    // 1) WebSocket Connection with Enhanced Data Handling
    // ------------------------------------------------------------------------
    const ws = new WebSocket("ws://127.0.0.1:5010");
    ws.binaryType = 'arraybuffer';

    let imageCounter = 0;
    let fpsCounter = 0;
    let lastFpsTime = Date.now();
    let frameCount = 0;

    ws.onopen = () => {
        console.log("WebSocket connected - Ikeda data stream active");
        updateStatusBar("CONNECTED", "WHITE");
    };

    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        updateStatusBar("CONNECTION ERROR", "RED");
    };

    ws.onmessage = (event) => {
        // Enhanced data handling with metadata support
        let byteArray = new Uint8Array(event.data);
        
        // Check if this is a metadata-packed message
        if (byteArray.length > 4) {
            try {
                // Try to read metadata size (first 4 bytes)
                const metadataSize = new DataView(byteArray.buffer, 0, 4).getUint32(0, true);
                
                if (metadataSize > 0 && metadataSize < byteArray.length) {
                    // Extract metadata
                    const metadataBytes = byteArray.slice(4, 4 + metadataSize);
                    const metadataStr = new TextDecoder().decode(metadataBytes);
                    const metadata = JSON.parse(metadataStr);
                    
                    // Extract image data
                    const imageData = byteArray.slice(4 + metadataSize);
                    
                    // Update data display with live analysis
                    updateDataDisplay(metadata.analysis);
                    
                    // Forward image to C++
                    let ptr = Module._malloc(imageData.length);
                    Module.HEAPU8.set(imageData, ptr);
                    Module.ccall('onImageReceived', null, ['number', 'number'], [ptr, imageData.length]);
                    Module._free(ptr);
                } else {
                    // Regular image data without metadata
                    let ptr = Module._malloc(byteArray.length);
                    Module.HEAPU8.set(byteArray, ptr);
                    Module.ccall('onImageReceived', null, ['number', 'number'], [ptr, byteArray.length]);
                    Module._free(ptr);
                }
            } catch (e) {
                // Fallback to regular image processing
                let ptr = Module._malloc(byteArray.length);
                Module.HEAPU8.set(byteArray, ptr);
                Module.ccall('onImageReceived', null, ['number', 'number'], [ptr, byteArray.length]);
                Module._free(ptr);
            }
        } else {
            // Regular image data
            let ptr = Module._malloc(byteArray.length);
            Module.HEAPU8.set(byteArray, ptr);
            Module.ccall('onImageReceived', null, ['number', 'number'], [ptr, byteArray.length]);
            Module._free(ptr);
        }

        // Update counters
        imageCounter++;
        frameCount++;
        document.getElementById('imageCounter').textContent = imageCounter;
        
        // Calculate FPS
        const now = Date.now();
        if (now - lastFpsTime >= 1000) {
            fpsCounter = frameCount;
            frameCount = 0;
            lastFpsTime = now;
            document.getElementById('fpsCounter').textContent = fpsCounter;
        }
    };

    // ------------------------------------------------------------------------
    // 2) Enhanced C++ Function Wrappers - Extended Ikeda Mode Support
    // ------------------------------------------------------------------------
    const setFadeFactor          = Module.cwrap('setFadeFactor', null, ['number']);
    const setImageSwitchInterval = Module.cwrap('setImageSwitchInterval', null, ['number']);
    const setTileFactor          = Module.cwrap('setTileFactor', null, ['number']);
    const setScrollSpeedX        = Module.cwrap('setScrollSpeedX', null, ['number']);
    const setScrollSpeedY        = Module.cwrap('setScrollSpeedY', null, ['number']);
    const setScrollOffsetX       = Module.cwrap('setScrollOffsetX', null, ['number']);
    const setScrollOffsetY       = Module.cwrap('setScrollOffsetY', null, ['number']);
    const setMaxUploadsPerFrame  = Module.cwrap('setMaxUploadsPerFrame', null, ['number']);
    const getBufferUsage         = Module.cwrap('getBufferUsage', 'number', []);

    // Extended Ikeda Mode Functions
    const setIkedaMode           = Module.cwrap('setIkedaMode', null, ['number']);
    const setIkedaThreshold      = Module.cwrap('setIkedaThreshold', null, ['number']);
    const setIkedaGridSize       = Module.cwrap('setIkedaGridSize', null, ['number']);
    const setIkedaDataIntensity  = Module.cwrap('setIkedaDataIntensity', null, ['number']);
    const setIkedaFrequency      = Module.cwrap('setIkedaFrequency', null, ['number']);
    const setIkedaScanSpeed      = Module.cwrap('setIkedaScanSpeed', null, ['number']);
    const setIkedaMatrixScale    = Module.cwrap('setIkedaMatrixScale', null, ['number']);
    const setIkedaPulseRate      = Module.cwrap('setIkedaPulseRate', null, ['number']);
    const setIkedaNoiseLevel     = Module.cwrap('setIkedaNoiseLevel', null, ['number']);
    const setIkedaStripWidth     = Module.cwrap('setIkedaStripWidth', null, ['number']);
    const setIkedaPhaseShift     = Module.cwrap('setIkedaPhaseShift', null, ['number']);
    const setIkedaQuantumLevels  = Module.cwrap('setIkedaQuantumLevels', null, ['number']);

    // ------------------------------------------------------------------------
    // 3) Enhanced UI Control System
    // ------------------------------------------------------------------------

    // Ikeda Mode Selection with Enhanced Controls
    const ikedaModeSelect = document.getElementById('ikedaMode');
    const modeNames = [
        'NORMAL', 'BLACK/WHITE', 'GRID', 'DATA', 'BINARY', 
        'FREQUENCY', 'SCAN', 'MATRIX', 'PULSE', 'NOISE',
        'STRIP', 'PHASE', 'QUANTUM'
    ];

    ikedaModeSelect.addEventListener('change', () => {
        const mode = parseInt(ikedaModeSelect.value);
        setIkedaMode(mode);
        updateModeDisplay(modeNames[mode]);
        showModeControls(mode);
        flashModeIndicator(modeNames[mode]);
    });

    // Core Ikeda Parameters
    setupSlider('ikedaThreshold', 'thresholdValue', setIkedaThreshold);
    setupSlider('ikedaGridSize', 'gridSizeValue', setIkedaGridSize);
    setupSlider('ikedaDataIntensity', 'dataIntensityValue', setIkedaDataIntensity);

    // Mode-Specific Parameters
    setupSlider('ikedaFrequency', 'frequencyValue', setIkedaFrequency);
    setupSlider('ikedaScanSpeed', 'scanSpeedValue', setIkedaScanSpeed);
    setupSlider('ikedaMatrixScale', 'matrixScaleValue', setIkedaMatrixScale);
    setupSlider('ikedaPulseRate', 'pulseRateValue', setIkedaPulseRate);
    setupSlider('ikedaNoiseLevel', 'noiseLevelValue', setIkedaNoiseLevel);
    setupSlider('ikedaStripWidth', 'stripWidthValue', setIkedaStripWidth);
    setupSlider('ikedaPhaseShift', 'phaseShiftValue', setIkedaPhaseShift);
    setupSlider('ikedaQuantumLevels', 'quantumLevelsValue', setIkedaQuantumLevels);

    // Original Controls
    setupSlider('fadeSlider', 'fadeValue', setFadeFactor);
    setupSlider('switchSlider', 'switchValue', setImageSwitchInterval);
    setupSlider('tileSlider', 'tileValue', setTileFactor);
    setupSlider('scrollSpeedX', 'scrollSpeedXVal', setScrollSpeedX);
    setupSlider('scrollSpeedY', 'scrollSpeedYVal', setScrollSpeedY);
    setupSlider('scrollOffsetX', 'scrollOffsetXVal', setScrollOffsetX);
    setupSlider('scrollOffsetY', 'scrollOffsetYVal', setScrollOffsetY);
    setupSlider('uploadsSlider', 'uploadsValue', setMaxUploadsPerFrame);

    // ------------------------------------------------------------------------
    // 4) Enhanced Keyboard Shortcuts - Exhibition Ready
    // ------------------------------------------------------------------------
    document.addEventListener('keydown', (event) => {
        const key = event.key.toLowerCase();
        
        // Mode selection shortcuts (1-9, 0, -, =)
        if (key >= '1' && key <= '9') {
            const mode = parseInt(key);
            selectMode(mode);
            event.preventDefault();
        } else if (key === '0') {
            selectMode(10); // STRIP
            event.preventDefault();
        } else if (key === '-') {
            selectMode(11); // PHASE
            event.preventDefault();
        } else if (key === '=') {
            selectMode(12); // QUANTUM
            event.preventDefault();
        }
        
        // Quick mode access
        switch (key) {
            case 'b': selectMode(1); break; // BLACK/WHITE
            case 'g': selectMode(2); break; // GRID  
            case 'd': selectMode(3); break; // DATA
            case 'f': selectMode(5); break; // FREQUENCY
            case 's': selectMode(6); break; // SCAN
            case 'm': selectMode(7); break; // MATRIX
            case 'p': selectMode(8); break; // PULSE
            case 'n': selectMode(9); break; // NOISE
            case 't': cycleThreshold(); break;
            case 'r': resetDefaults(); break;
            case 'escape': toggleFullscreen(); break;
        }
    });

    // ------------------------------------------------------------------------
    // 5) Enhanced System Functions
    // ------------------------------------------------------------------------

    function setupSlider(sliderId, valueId, setterFunction) {
        const slider = document.getElementById(sliderId);
        const valueDisplay = document.getElementById(valueId);
        
        if (slider && valueDisplay) {
            slider.addEventListener('input', () => {
                const value = parseFloat(slider.value);
                setterFunction(value);
                valueDisplay.textContent = value.toFixed(2);
            });
        }
    }

    function selectMode(mode) {
        ikedaModeSelect.value = mode;
        setIkedaMode(mode);
        updateModeDisplay(modeNames[mode]);
        showModeControls(mode);
        flashModeIndicator(modeNames[mode]);
    }

    function showModeControls(mode) {
        // Hide all mode controls
        document.querySelectorAll('.mode-controls').forEach(control => {
            control.classList.remove('active');
        });
        
        // Show relevant controls
        const controlMap = {
            5: 'frequencyControls',
            6: 'scanControls', 
            7: 'matrixControls',
            8: 'pulseControls',
            9: 'noiseControls',
            10: 'stripControls',
            11: 'phaseControls',
            12: 'quantumControls'
        };
        
        const controlId = controlMap[mode];
        if (controlId) {
            const control = document.getElementById(controlId);
            if (control) control.classList.add('active');
        }
    }

    function updateModeDisplay(modeName) {
        document.getElementById('currentMode').textContent = modeName;
    }

    function flashModeIndicator(modeName) {
        const indicator = document.getElementById('modeIndicator');
        indicator.textContent = modeName;
        indicator.classList.add('mode-flash');
        setTimeout(() => indicator.classList.remove('mode-flash'), 1000);
    }

    function cycleThreshold() {
        const slider = document.getElementById('ikedaThreshold');
        const thresholds = [0.1, 0.3, 0.5, 0.7, 0.9];
        const current = parseFloat(slider.value);
        let nextIndex = 0;
        
        for (let i = 0; i < thresholds.length; i++) {
            if (Math.abs(current - thresholds[i]) < 0.05) {
                nextIndex = (i + 1) % thresholds.length;
                break;
            }
        }
        
        const newValue = thresholds[nextIndex];
        slider.value = newValue;
        setIkedaThreshold(newValue);
        document.getElementById('thresholdValue').textContent = newValue.toFixed(2);
    }

    function resetDefaults() {
        // Reset to default BLACK/WHITE mode
        selectMode(1);
        
        // Reset core parameters
        setSliderValue('ikedaThreshold', 0.5);
        setSliderValue('ikedaGridSize', 32);
        setSliderValue('ikedaDataIntensity', 0.5);
        
        // Reset animation parameters
        setSliderValue('fadeSlider', 0.5);
        setSliderValue('switchSlider', 0.33);
        setSliderValue('tileSlider', 3);
        
        // Reset motion parameters
        setSliderValue('scrollSpeedX', 0.1);
        setSliderValue('scrollSpeedY', 0.0);
        setSliderValue('scrollOffsetX', 0.1);
        setSliderValue('scrollOffsetY', 0.0);
        
        flashModeIndicator('RESET');
    }

    function setSliderValue(sliderId, value) {
        const slider = document.getElementById(sliderId);
        if (slider) {
            slider.value = value;
            slider.dispatchEvent(new Event('input'));
        }
    }

    function toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }

    function updateStatusBar(status, color) {
        const statusElement = document.getElementById('connectionStatus');
        statusElement.textContent = status;
        statusElement.style.color = color || '#FFFFFF';
    }

    function updateDataDisplay(analysis) {
        if (!analysis) return;
        
        document.getElementById('dataLuminance').textContent = 
            analysis.luminance ? analysis.luminance.toFixed(3) : '--';
        document.getElementById('dataEntropy').textContent = 
            analysis.entropy ? analysis.entropy.toFixed(3) : '--';
        document.getElementById('dataVariance').textContent = 
            analysis.variance ? analysis.variance.toFixed(3) : '--';
        document.getElementById('dataEdgeDensity').textContent = 
            analysis.edge_density ? analysis.edge_density.toFixed(3) : '--';
        document.getElementById('dataFreqRatio').textContent = 
            analysis.freq_ratio ? analysis.freq_ratio.toFixed(3) : '--';
        document.getElementById('dataCompression').textContent = 
            analysis.compression ? analysis.compression.toFixed(2) : '--';
        document.getElementById('dataTimestamp').textContent = 
            new Date().toLocaleTimeString();
    }

    // ------------------------------------------------------------------------
    // 6) System Controls and Buffer Management
    // ------------------------------------------------------------------------
    document.getElementById('updateBufferUsage').addEventListener('click', () => {
        const usage = getBufferUsage();
        document.getElementById('bufferUsageLabel').textContent = `Buffer: ${usage}`;
    });

    document.getElementById('resetSystem').addEventListener('click', () => {
        resetDefaults();
    });

    // ------------------------------------------------------------------------
    // 7) Initialize System
    // ------------------------------------------------------------------------
    
    // Set initial mode to BLACK/WHITE
    selectMode(1);
    
    // Update status
    updateStatusBar("INITIALIZING...", "YELLOW");
    
    console.log("Enhanced Ikeda control system initialized with 13 visual modes");
    console.log("Keyboard shortcuts active - Press B/G/D/F/S/M/P/N for quick mode switching");
};

// Enhanced error handling for WebAssembly initialization
Module['onAbort'] = (what) => {
    console.error("WebAssembly module aborted:", what);
    document.getElementById('connectionStatus').textContent = "WASM ERROR";
};

console.log("Enhanced Ikeda control system loading..."); 