// Enhanced Ikeda-inspired control system for ImageFlasherWGPU
// Handles new visual modes, data analysis, and precise control interface

Module['onRuntimeInitialized'] = () => {
    console.log("WASM runtime initialized. Setting up Ikeda control system...");

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
        try {
            // Handle enhanced data format with metadata
            const data = new Uint8Array(event.data);
            
            // Check if this is a packed data format
            if (data.length > 8) {
                const view = new DataView(event.data);
                const metadataSize = view.getUint32(0, true); // little endian
                
                if (metadataSize > 0 && metadataSize < data.length) {
                    // Extract metadata
                    const metadataBytes = data.slice(4, 4 + metadataSize);
                    const metadataJson = new TextDecoder().decode(metadataBytes);
                    
                    try {
                        const metadata = JSON.parse(metadataJson);
                        updateDataDisplay(metadata.analysis);
                        
                        // Extract image data
                        const imageData = data.slice(4 + metadataSize);
                        processImageData(imageData);
                    } catch (parseError) {
                        console.warn("Metadata parse error, using raw image data");
                        processImageData(data);
                    }
                } else {
                    processImageData(data);
                }
            } else {
                processImageData(data);
            }
            
            imageCounter++;
            frameCount++;
            updateCounters();
            
        } catch (error) {
            console.error("Error processing image data:", error);
        }
    };

    function processImageData(imageData) {
        // Allocate memory in WASM heap
        let ptr = Module._malloc(imageData.length);
        Module.HEAPU8.set(imageData, ptr);

        // Forward to C++
        Module.ccall('onImageReceived', null, ['number', 'number'], [ptr, imageData.length]);

        // Free the temporary buffer
        Module._free(ptr);
    }

    // ------------------------------------------------------------------------
    // 2) Enhanced Control System with Ikeda Functions
    // ------------------------------------------------------------------------

    // Create cwrap references for all functions
    const setFadeFactor          = Module.cwrap('setFadeFactor', null, ['number']);
    const setImageSwitchInterval = Module.cwrap('setImageSwitchInterval', null, ['number']);
    const getBufferUsage         = Module.cwrap('getBufferUsage', 'number', []);
    const getRingBufferSize      = Module.cwrap('getRingBufferSize', 'number', []);
    const setMaxUploadsPerFrame  = Module.cwrap('setMaxUploadsPerFrame', null, ['number']);
    const setScrollingSpeed      = Module.cwrap('setScrollingSpeed', null, ['number', 'number']);
    const setScrollingOffset     = Module.cwrap('setScrollingOffset', null, ['number', 'number']);
    const setTileFactor          = Module.cwrap('setTileFactor', null, ['number']);

    // New Ikeda functions (these would need to be implemented in the C++ code)
    const setIkedaMode           = Module.cwrap('setIkedaMode', null, ['number']);
    const setIkedaThreshold      = Module.cwrap('setIkedaThreshold', null, ['number']);
    const setIkedaGridSize       = Module.cwrap('setIkedaGridSize', null, ['number']);
    const setIkedaDataIntensity  = Module.cwrap('setIkedaDataIntensity', null, ['number']);
    const getImageAverageLuminance = Module.cwrap('getImageAverageLuminance', 'number', []);
    const getImageEntropy        = Module.cwrap('getImageEntropy', 'number', []);
    const getImageVariance       = Module.cwrap('getImageVariance', 'number', []);

    // Get DOM elements
    const elements = {
        // Ikeda controls
        ikedaMode: document.getElementById('ikedaMode'),
        ikedaThreshold: document.getElementById('ikedaThreshold'),
        ikedaGridSize: document.getElementById('ikedaGridSize'),
        ikedaDataIntensity: document.getElementById('ikedaDataIntensity'),
        
        // Value displays
        thresholdValue: document.getElementById('thresholdValue'),
        gridSizeValue: document.getElementById('gridSizeValue'),
        dataIntensityValue: document.getElementById('dataIntensityValue'),
        
        // Original controls
        fadeSlider: document.getElementById('fadeSlider'),
        fadeValue: document.getElementById('fadeValue'),
        switchSlider: document.getElementById('switchSlider'),
        switchValue: document.getElementById('switchValue'),
        tileSlider: document.getElementById('tileSlider'),
        tileValue: document.getElementById('tileValue'),
        uploadsSlider: document.getElementById('uploadsSlider'),
        uploadsValue: document.getElementById('uploadsValue'),
        
        // Scrolling controls
        scrollSpeedX: document.getElementById('scrollSpeedX'),
        scrollSpeedXVal: document.getElementById('scrollSpeedXVal'),
        scrollSpeedY: document.getElementById('scrollSpeedY'),
        scrollSpeedYVal: document.getElementById('scrollSpeedYVal'),
        scrollOffsetX: document.getElementById('scrollOffsetX'),
        scrollOffsetXVal: document.getElementById('scrollOffsetXVal'),
        scrollOffsetY: document.getElementById('scrollOffsetY'),
        scrollOffsetYVal: document.getElementById('scrollOffsetYVal'),
        
        // Buttons
        updateBufferUsageBtn: document.getElementById('updateBufferUsage'),
        resetSystemBtn: document.getElementById('resetSystem'),
        
        // Status elements
        bufferUsageLabel: document.getElementById('bufferUsageLabel'),
        fpsCounter: document.getElementById('fpsCounter'),
        currentMode: document.getElementById('currentMode'),
        imageCounter: document.getElementById('imageCounter'),
        modeIndicator: document.getElementById('modeIndicator'),
        
        // Data display
        dataLuminance: document.getElementById('dataLuminance'),
        dataEntropy: document.getElementById('dataEntropy'),
        dataVariance: document.getElementById('dataVariance'),
        dataEdgeDensity: document.getElementById('dataEdgeDensity'),
        dataFreqRatio: document.getElementById('dataFreqRatio'),
        dataCompression: document.getElementById('dataCompression'),
        dataTimestamp: document.getElementById('dataTimestamp')
    };

    // ------------------------------------------------------------------------
    // 3) Ikeda Mode Control
    // ------------------------------------------------------------------------

    const modeNames = ['NORMAL', 'BLACK/WHITE', 'GRID', 'DATA', 'BINARY'];

    elements.ikedaMode.addEventListener('change', () => {
        const mode = parseInt(elements.ikedaMode.value);
        setIkedaMode(mode);
        
        const modeName = modeNames[mode] || 'UNKNOWN';
        elements.currentMode.textContent = modeName;
        
        // Flash mode indicator
        elements.modeIndicator.textContent = modeName;
        elements.modeIndicator.classList.add('mode-flash');
        setTimeout(() => {
            elements.modeIndicator.classList.remove('mode-flash');
        }, 1000);
        
        console.log(`Ikeda mode changed to: ${modeName}`);
    });

    // Threshold control
    elements.ikedaThreshold.addEventListener('input', () => {
        const val = parseFloat(elements.ikedaThreshold.value);
        elements.thresholdValue.textContent = val.toFixed(2);
        setIkedaThreshold(val);
    });

    // Grid size control
    elements.ikedaGridSize.addEventListener('input', () => {
        const val = parseInt(elements.ikedaGridSize.value);
        elements.gridSizeValue.textContent = val;
        setIkedaGridSize(val);
    });

    // Data intensity control
    elements.ikedaDataIntensity.addEventListener('input', () => {
        const val = parseFloat(elements.ikedaDataIntensity.value);
        elements.dataIntensityValue.textContent = val.toFixed(2);
        setIkedaDataIntensity(val);
    });

    // ------------------------------------------------------------------------
    // 4) Original Controls (Enhanced)
    // ------------------------------------------------------------------------

    elements.fadeSlider.addEventListener('input', () => {
        const val = parseFloat(elements.fadeSlider.value);
        elements.fadeValue.textContent = val.toFixed(2);
        setFadeFactor(val);
    });

    elements.switchSlider.addEventListener('input', () => {
        const val = parseFloat(elements.switchSlider.value);
        elements.switchValue.textContent = val.toFixed(2);
        setImageSwitchInterval(val);
    });

    elements.tileSlider.addEventListener('input', () => {
        const val = parseInt(elements.tileSlider.value);
        elements.tileValue.textContent = val;
        setTileFactor(val);
    });

    elements.uploadsSlider.addEventListener('input', () => {
        const val = parseInt(elements.uploadsSlider.value);
        elements.uploadsValue.textContent = val;
        setMaxUploadsPerFrame(val);
    });

    // Scrolling controls
    elements.scrollSpeedX.addEventListener('input', () => {
        const val = parseFloat(elements.scrollSpeedX.value);
        elements.scrollSpeedXVal.textContent = val.toFixed(2);
        const speedY = parseFloat(elements.scrollSpeedY.value);
        setScrollingSpeed(val, speedY);
    });

    elements.scrollSpeedY.addEventListener('input', () => {
        const val = parseFloat(elements.scrollSpeedY.value);
        elements.scrollSpeedYVal.textContent = val.toFixed(2);
        const speedX = parseFloat(elements.scrollSpeedX.value);
        setScrollingSpeed(speedX, val);
    });

    elements.scrollOffsetX.addEventListener('input', () => {
        const val = parseFloat(elements.scrollOffsetX.value);
        elements.scrollOffsetXVal.textContent = val.toFixed(2);
        const oy = parseFloat(elements.scrollOffsetY.value);
        setScrollingOffset(val, oy);
    });

    elements.scrollOffsetY.addEventListener('input', () => {
        const val = parseFloat(elements.scrollOffsetY.value);
        elements.scrollOffsetYVal.textContent = val.toFixed(2);
        const ox = parseFloat(elements.scrollOffsetX.value);
        setScrollingOffset(ox, val);
    });

    // Buffer usage button
    elements.updateBufferUsageBtn.addEventListener('click', () => {
        const usage = getBufferUsage();
        const capacity = getRingBufferSize();
        elements.bufferUsageLabel.textContent = `Buffer: ${usage}/${capacity}`;
    });

    // Reset system button
    elements.resetSystemBtn.addEventListener('click', () => {
        // Reset all controls to defaults
        elements.ikedaMode.value = "1";
        elements.ikedaThreshold.value = "0.5";
        elements.ikedaGridSize.value = "32";
        elements.ikedaDataIntensity.value = "0.5";
        elements.fadeSlider.value = "0.5";
        elements.switchSlider.value = "0.33";
        elements.tileSlider.value = "3";
        elements.uploadsSlider.value = "0";
        elements.scrollSpeedX.value = "0.1";
        elements.scrollSpeedY.value = "0.00";
        elements.scrollOffsetX.value = "0.10";
        elements.scrollOffsetY.value = "0.00";
        
        // Trigger all change events
        elements.ikedaMode.dispatchEvent(new Event('change'));
        elements.ikedaThreshold.dispatchEvent(new Event('input'));
        elements.ikedaGridSize.dispatchEvent(new Event('input'));
        elements.ikedaDataIntensity.dispatchEvent(new Event('input'));
        elements.fadeSlider.dispatchEvent(new Event('input'));
        elements.switchSlider.dispatchEvent(new Event('input'));
        elements.tileSlider.dispatchEvent(new Event('input'));
        elements.uploadsSlider.dispatchEvent(new Event('input'));
        elements.scrollSpeedX.dispatchEvent(new Event('input'));
        elements.scrollSpeedY.dispatchEvent(new Event('input'));
        elements.scrollOffsetX.dispatchEvent(new Event('input'));
        elements.scrollOffsetY.dispatchEvent(new Event('input'));
        
        console.log("System reset to defaults");
    });

    // ------------------------------------------------------------------------
    // 5) Keyboard Shortcuts (Ikeda Exhibition Mode)
    // ------------------------------------------------------------------------

    document.addEventListener('keydown', (event) => {
        switch(event.key) {
            case '1':
            case '2':
            case '3':
            case '4':
                const mode = event.key;
                elements.ikedaMode.value = (parseInt(mode) - 1).toString();
                elements.ikedaMode.dispatchEvent(new Event('change'));
                break;
                
            case 'b':
            case 'B':
                elements.ikedaMode.value = "1"; // Black/White mode
                elements.ikedaMode.dispatchEvent(new Event('change'));
                break;
                
            case 'g':
            case 'G':
                // Cycle through grid sizes
                const currentGrid = parseInt(elements.ikedaGridSize.value);
                const gridSizes = [8, 16, 32, 64, 128];
                const currentIndex = gridSizes.indexOf(currentGrid);
                const nextGrid = gridSizes[(currentIndex + 1) % gridSizes.length];
                elements.ikedaGridSize.value = nextGrid;
                elements.ikedaGridSize.dispatchEvent(new Event('input'));
                break;
                
            case 't':
            case 'T':
                // Toggle between high and low threshold
                const currentThreshold = parseFloat(elements.ikedaThreshold.value);
                const newThreshold = currentThreshold > 0.5 ? 0.2 : 0.8;
                elements.ikedaThreshold.value = newThreshold;
                elements.ikedaThreshold.dispatchEvent(new Event('input'));
                break;
                
            case 'r':
            case 'R':
                elements.resetSystemBtn.click();
                break;
                
            case 'Escape':
                toggleFullscreen();
                break;
        }
    });

    // ------------------------------------------------------------------------
    // 6) Data Display Updates
    // ------------------------------------------------------------------------

    function updateDataDisplay(analysisData) {
        if (!analysisData) return;
        
        elements.dataLuminance.textContent = analysisData.mean_luminance?.toFixed(1) || '---';
        elements.dataEntropy.textContent = analysisData.entropy?.toFixed(2) || '---';
        elements.dataVariance.textContent = analysisData.variance?.toFixed(0) || '---';
        elements.dataEdgeDensity.textContent = analysisData.edge_density?.toFixed(3) || '---';
        elements.dataFreqRatio.textContent = analysisData.high_freq_ratio?.toFixed(2) || '---';
        elements.dataCompression.textContent = analysisData.estimated_compression?.toFixed(2) || '---';
        
        if (analysisData.timestamp) {
            const date = new Date(analysisData.timestamp * 1000);
            elements.dataTimestamp.textContent = date.toLocaleTimeString();
        }
    }

    function updateCounters() {
        elements.imageCounter.textContent = imageCounter;
        
        // Update FPS counter
        const now = Date.now();
        if (now - lastFpsTime >= 1000) {
            fpsCounter = Math.round(frameCount * 1000 / (now - lastFpsTime));
            elements.fpsCounter.textContent = fpsCounter;
            frameCount = 0;
            lastFpsTime = now;
        }
    }

    function updateStatusBar(message, color) {
        const statusElements = document.querySelectorAll('#statusBar span');
        if (statusElements.length > 0) {
            statusElements[statusElements.length - 1].textContent = message;
            statusElements[statusElements.length - 1].style.color = color || '#FFFFFF';
        }
    }

    function toggleFullscreen() {
        const controls = document.getElementById('controls');
        const dataDisplay = document.getElementById('dataDisplay');
        const shortcuts = document.getElementById('shortcuts');
        const statusBar = document.getElementById('statusBar');
        
        const isHidden = controls.style.display === 'none';
        
        const displayValue = isHidden ? 'block' : 'none';
        controls.style.display = displayValue;
        dataDisplay.style.display = displayValue;
        shortcuts.style.display = displayValue;
        statusBar.style.display = displayValue;
        
        console.log(isHidden ? "UI shown" : "UI hidden (exhibition mode)");
    }

    // ------------------------------------------------------------------------
    // 7) Initialize Default Values
    // ------------------------------------------------------------------------
    
    // Set initial values and trigger events
    setTimeout(() => {
        elements.ikedaMode.dispatchEvent(new Event('change'));
        elements.ikedaThreshold.dispatchEvent(new Event('input'));
        elements.ikedaGridSize.dispatchEvent(new Event('input'));
        elements.ikedaDataIntensity.dispatchEvent(new Event('input'));
        elements.fadeSlider.dispatchEvent(new Event('input'));
        elements.switchSlider.dispatchEvent(new Event('input'));
        elements.tileSlider.dispatchEvent(new Event('input'));
        elements.uploadsSlider.dispatchEvent(new Event('input'));
        elements.scrollSpeedX.dispatchEvent(new Event('input'));
        elements.scrollSpeedY.dispatchEvent(new Event('input'));
        elements.scrollOffsetX.dispatchEvent(new Event('input'));
        elements.scrollOffsetY.dispatchEvent(new Event('input'));
        
        console.log("Ikeda control system initialized");
        updateStatusBar("DATA.MATRIX READY", "#FFFFFF");
    }, 100);

    // Update buffer usage periodically
    setInterval(() => {
        if (getBufferUsage && getRingBufferSize) {
            const usage = getBufferUsage();
            const capacity = getRingBufferSize();
            elements.bufferUsageLabel.textContent = `Buffer: ${usage}/${capacity}`;
        }
    }, 5000);

    console.log("Ikeda WebSocket + UI control system ready");
}; 