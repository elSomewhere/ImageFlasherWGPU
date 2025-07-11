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
    // 2) Enhanced Control System with Extended Ikeda Functions
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

    // Core Ikeda functions
    const setIkedaMode           = Module.cwrap('setIkedaMode', null, ['number']);
    const setIkedaThreshold      = Module.cwrap('setIkedaThreshold', null, ['number']);
    const setIkedaGridSize       = Module.cwrap('setIkedaGridSize', null, ['number']);
    const setIkedaDataIntensity  = Module.cwrap('setIkedaDataIntensity', null, ['number']);
    
    // Extended Ikeda functions for new modes
    const setIkedaFrequency      = Module.cwrap('setIkedaFrequency', null, ['number']);
    const setIkedaPhaseShift     = Module.cwrap('setIkedaPhaseShift', null, ['number']);
    const setIkedaNoiseLevel     = Module.cwrap('setIkedaNoiseLevel', null, ['number']);
    const setIkedaStripWidth     = Module.cwrap('setIkedaStripWidth', null, ['number']);
    const setIkedaQuantumLevels  = Module.cwrap('setIkedaQuantumLevels', null, ['number']);
    const setIkedaScanSpeed      = Module.cwrap('setIkedaScanSpeed', null, ['number']);
    const setIkedaMatrixScale    = Module.cwrap('setIkedaMatrixScale', null, ['number']);
    const setIkedaPulseRate      = Module.cwrap('setIkedaPulseRate', null, ['number']);
    
    // Data analysis functions
    const getImageAverageLuminance = Module.cwrap('getImageAverageLuminance', 'number', []);
    const getImageEntropy        = Module.cwrap('getImageEntropy', 'number', []);
    const getImageVariance       = Module.cwrap('getImageVariance', 'number', []);

    // Get DOM elements
    const elements = {
        // Ikeda mode controls
        ikedaMode: document.getElementById('ikedaMode'),
        ikedaThreshold: document.getElementById('ikedaThreshold'),
        ikedaGridSize: document.getElementById('ikedaGridSize'),
        ikedaDataIntensity: document.getElementById('ikedaDataIntensity'),
        
        // Extended mode parameters
        ikedaFrequency: document.getElementById('ikedaFrequency'),
        ikedaScanSpeed: document.getElementById('ikedaScanSpeed'),
        ikedaMatrixScale: document.getElementById('ikedaMatrixScale'),
        ikedaPulseRate: document.getElementById('ikedaPulseRate'),
        ikedaNoiseLevel: document.getElementById('ikedaNoiseLevel'),
        ikedaStripWidth: document.getElementById('ikedaStripWidth'),
        ikedaPhaseShift: document.getElementById('ikedaPhaseShift'),
        ikedaQuantumLevels: document.getElementById('ikedaQuantumLevels'),
        
        // Value displays
        thresholdValue: document.getElementById('thresholdValue'),
        gridSizeValue: document.getElementById('gridSizeValue'),
        dataIntensityValue: document.getElementById('dataIntensityValue'),
        frequencyValue: document.getElementById('frequencyValue'),
        scanSpeedValue: document.getElementById('scanSpeedValue'),
        matrixScaleValue: document.getElementById('matrixScaleValue'),
        pulseRateValue: document.getElementById('pulseRateValue'),
        noiseLevelValue: document.getElementById('noiseLevelValue'),
        stripWidthValue: document.getElementById('stripWidthValue'),
        phaseShiftValue: document.getElementById('phaseShiftValue'),
        quantumLevelsValue: document.getElementById('quantumLevelsValue'),
        
        // Core parameter rows
        thresholdRow: document.getElementById('thresholdRow'),
        gridSizeRow: document.getElementById('gridSizeRow'),
        dataIntensityRow: document.getElementById('dataIntensityRow'),
        
        // Mode-specific control panels
        frequencyControls: document.getElementById('frequencyControls'),
        scanControls: document.getElementById('scanControls'),
        matrixControls: document.getElementById('matrixControls'),
        pulseControls: document.getElementById('pulseControls'),
        noiseControls: document.getElementById('noiseControls'),
        stripControls: document.getElementById('stripControls'),
        phaseControls: document.getElementById('phaseControls'),
        quantumControls: document.getElementById('quantumControls'),
        
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
        connectionStatus: document.getElementById('connectionStatus'),
        
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
    // 3) Enhanced Ikeda Mode Control
    // ------------------------------------------------------------------------

    const modeNames = [
        'NORMAL', 'BLACK/WHITE', 'GRID', 'DATA', 'BINARY',
        'FREQUENCY', 'SCAN', 'MATRIX', 'PULSE', 'NOISE',
        'STRIP', 'PHASE', 'QUANTUM'
    ];

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
        
        // Update UI based on mode
        updateUIForMode(mode);
        
        console.log(`Ikeda mode changed to: ${modeName}`);
    });

    // Core parameter controls
    elements.ikedaThreshold.addEventListener('input', () => {
        const val = parseFloat(elements.ikedaThreshold.value);
        elements.thresholdValue.textContent = val.toFixed(2);
        setIkedaThreshold(val);
    });

    elements.ikedaGridSize.addEventListener('input', () => {
        const val = parseInt(elements.ikedaGridSize.value);
        elements.gridSizeValue.textContent = val;
        setIkedaGridSize(val);
    });

    elements.ikedaDataIntensity.addEventListener('input', () => {
        const val = parseFloat(elements.ikedaDataIntensity.value);
        elements.dataIntensityValue.textContent = val.toFixed(2);
        setIkedaDataIntensity(val);
    });

    // ------------------------------------------------------------------------
    // 4) Extended Mode Parameter Controls
    // ------------------------------------------------------------------------

    // Frequency mode controls
    elements.ikedaFrequency.addEventListener('input', () => {
        const val = parseFloat(elements.ikedaFrequency.value);
        elements.frequencyValue.textContent = val.toFixed(1);
        setIkedaFrequency(val);
    });

    // Scan mode controls
    elements.ikedaScanSpeed.addEventListener('input', () => {
        const val = parseFloat(elements.ikedaScanSpeed.value);
        elements.scanSpeedValue.textContent = val.toFixed(1);
        setIkedaScanSpeed(val);
    });

    // Matrix mode controls
    elements.ikedaMatrixScale.addEventListener('input', () => {
        const val = parseFloat(elements.ikedaMatrixScale.value);
        elements.matrixScaleValue.textContent = val.toFixed(1);
        setIkedaMatrixScale(val);
    });

    // Pulse mode controls
    elements.ikedaPulseRate.addEventListener('input', () => {
        const val = parseFloat(elements.ikedaPulseRate.value);
        elements.pulseRateValue.textContent = val.toFixed(1);
        setIkedaPulseRate(val);
    });

    // Noise mode controls
    elements.ikedaNoiseLevel.addEventListener('input', () => {
        const val = parseFloat(elements.ikedaNoiseLevel.value);
        elements.noiseLevelValue.textContent = val.toFixed(2);
        setIkedaNoiseLevel(val);
    });

    // Strip mode controls
    elements.ikedaStripWidth.addEventListener('input', () => {
        const val = parseFloat(elements.ikedaStripWidth.value);
        elements.stripWidthValue.textContent = val.toFixed(2);
        setIkedaStripWidth(val);
    });

    // Phase mode controls
    elements.ikedaPhaseShift.addEventListener('input', () => {
        const val = parseFloat(elements.ikedaPhaseShift.value);
        elements.phaseShiftValue.textContent = val.toFixed(2);
        setIkedaPhaseShift(val);
    });

    // Quantum mode controls
    elements.ikedaQuantumLevels.addEventListener('input', () => {
        const val = parseInt(elements.ikedaQuantumLevels.value);
        elements.quantumLevelsValue.textContent = val;
        setIkedaQuantumLevels(val);
    });

    // ------------------------------------------------------------------------
    // 5) Original Control Event Listeners (Enhanced)
    // ------------------------------------------------------------------------

    elements.fadeSlider.addEventListener('input', () => {
        const val = parseFloat(elements.fadeSlider.value);
        elements.fadeValue.textContent = val.toFixed(2);
        setFadeFactor(val);
    });

    elements.switchSlider.addEventListener('input', () => {
        const val = parseFloat(elements.switchSlider.value);
        elements.switchValue.textContent = val.toFixed(4);
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
        const sy = parseFloat(elements.scrollSpeedY.value);
        setScrollingSpeed(val, sy);
    });

    elements.scrollSpeedY.addEventListener('input', () => {
        const val = parseFloat(elements.scrollSpeedY.value);
        elements.scrollSpeedYVal.textContent = val.toFixed(2);
        const sx = parseFloat(elements.scrollSpeedX.value);
        setScrollingSpeed(sx, val);
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
        resetAllControls();
    });

    // ------------------------------------------------------------------------
    // 6) Enhanced Keyboard Shortcuts
    // ------------------------------------------------------------------------

    document.addEventListener('keydown', (event) => {
        if (event.target.tagName === 'INPUT') return; // Don't interfere with input fields
        
        switch(event.key) {
            // Mode switching (1-9, 0, -, =)
            case '1': switchToMode(1); break;
            case '2': switchToMode(2); break;
            case '3': switchToMode(3); break;
            case '4': switchToMode(4); break;
            case '5': switchToMode(5); break;
            case '6': switchToMode(6); break;
            case '7': switchToMode(7); break;
            case '8': switchToMode(8); break;
            case '9': switchToMode(9); break;
            case '0': switchToMode(10); break;
            case '-': switchToMode(11); break;
            case '=': switchToMode(12); break;
            
            // Mode shortcuts
            case 'b':
            case 'B':
                switchToMode(1); // Black/White
                break;
            case 'g':
            case 'G':
                switchToMode(2); // Grid
                break;
            case 'd':
            case 'D':
                switchToMode(3); // Data
                break;
            case 'f':
            case 'F':
                switchToMode(5); // Frequency
                break;
            case 's':
            case 'S':
                switchToMode(6); // Scan
                break;
            case 'm':
            case 'M':
                switchToMode(7); // Matrix
                break;
            case 'p':
            case 'P':
                switchToMode(8); // Pulse
                break;
            case 'n':
            case 'N':
                switchToMode(9); // Noise
                break;
                
            // Parameter adjustments
            case 't':
            case 'T':
                cycleThreshold();
                break;
            case 'r':
            case 'R':
                resetAllControls();
                break;
                
            // Fullscreen
            case 'Escape':
                toggleFullscreen();
                break;
        }
        
        event.preventDefault();
    });

    // ------------------------------------------------------------------------
    // 7) UI Management Functions
    // ------------------------------------------------------------------------

    function updateUIForMode(mode) {
        // Hide all core parameter rows by default
        const allCoreRows = [elements.thresholdRow, elements.gridSizeRow, elements.dataIntensityRow];
        allCoreRows.forEach(row => {
            if (row) row.style.display = 'none';
        });
        
        // Hide all mode-specific controls
        const allModeControls = [
            elements.frequencyControls, elements.scanControls, elements.matrixControls,
            elements.pulseControls, elements.noiseControls, elements.stripControls,
            elements.phaseControls, elements.quantumControls
        ];
        
        allModeControls.forEach(control => {
            if (control) control.classList.remove('active');
        });
        
        // Show appropriate core parameters based on mode
        switch(mode) {
            case 0: // NORMAL - no core parameters
                break;
            case 1: // BLACK/WHITE - only threshold
                if (elements.thresholdRow) elements.thresholdRow.style.display = 'flex';
                break;
            case 2: // GRID - threshold and grid size
                if (elements.thresholdRow) elements.thresholdRow.style.display = 'flex';
                if (elements.gridSizeRow) elements.gridSizeRow.style.display = 'flex';
                break;
            case 3: // DATA - threshold and data intensity
                if (elements.thresholdRow) elements.thresholdRow.style.display = 'flex';
                if (elements.dataIntensityRow) elements.dataIntensityRow.style.display = 'flex';
                break;
            case 4: // BINARY - only threshold
                if (elements.thresholdRow) elements.thresholdRow.style.display = 'flex';
                break;
            case 5: // FREQUENCY - threshold and data intensity
                if (elements.thresholdRow) elements.thresholdRow.style.display = 'flex';
                if (elements.dataIntensityRow) elements.dataIntensityRow.style.display = 'flex';
                if (elements.frequencyControls) elements.frequencyControls.classList.add('active');
                break;
            case 6: // SCAN - threshold and data intensity
                if (elements.thresholdRow) elements.thresholdRow.style.display = 'flex';
                if (elements.dataIntensityRow) elements.dataIntensityRow.style.display = 'flex';
                if (elements.scanControls) elements.scanControls.classList.add('active');
                break;
            case 7: // MATRIX - threshold and data intensity
                if (elements.thresholdRow) elements.thresholdRow.style.display = 'flex';
                if (elements.dataIntensityRow) elements.dataIntensityRow.style.display = 'flex';
                if (elements.matrixControls) elements.matrixControls.classList.add('active');
                break;
            case 8: // PULSE - threshold and data intensity
                if (elements.thresholdRow) elements.thresholdRow.style.display = 'flex';
                if (elements.dataIntensityRow) elements.dataIntensityRow.style.display = 'flex';
                if (elements.pulseControls) elements.pulseControls.classList.add('active');
                break;
            case 9: // NOISE - threshold and data intensity
                if (elements.thresholdRow) elements.thresholdRow.style.display = 'flex';
                if (elements.dataIntensityRow) elements.dataIntensityRow.style.display = 'flex';
                if (elements.noiseControls) elements.noiseControls.classList.add('active');
                break;
            case 10: // STRIP - only threshold (no data intensity overlay)
                if (elements.thresholdRow) elements.thresholdRow.style.display = 'flex';
                if (elements.stripControls) elements.stripControls.classList.add('active');
                break;
            case 11: // PHASE - threshold and data intensity
                if (elements.thresholdRow) elements.thresholdRow.style.display = 'flex';
                if (elements.dataIntensityRow) elements.dataIntensityRow.style.display = 'flex';
                if (elements.phaseControls) elements.phaseControls.classList.add('active');
                break;
            case 12: // QUANTUM - only threshold (uses its own quantization)
                if (elements.thresholdRow) elements.thresholdRow.style.display = 'flex';
                if (elements.quantumControls) elements.quantumControls.classList.add('active');
                break;
        }
    }

    function switchToMode(mode) {
        elements.ikedaMode.value = mode;
        elements.ikedaMode.dispatchEvent(new Event('change'));
    }

    function cycleThreshold() {
        const currentThreshold = parseFloat(elements.ikedaThreshold.value);
        const thresholds = [0.1, 0.3, 0.5, 0.7, 0.9];
        const currentIndex = thresholds.findIndex(t => Math.abs(t - currentThreshold) < 0.01);
        const nextIndex = (currentIndex + 1) % thresholds.length;
        
        elements.ikedaThreshold.value = thresholds[nextIndex];
        elements.ikedaThreshold.dispatchEvent(new Event('input'));
    }

    function resetAllControls() {
        // Reset all controls to defaults
        elements.ikedaMode.value = "1";
        elements.ikedaThreshold.value = "0.5";
        elements.ikedaGridSize.value = "32";
        elements.ikedaDataIntensity.value = "0.5";
        elements.ikedaFrequency.value = "3";
        elements.ikedaScanSpeed.value = "0.5";
        elements.ikedaMatrixScale.value = "1";
        elements.ikedaPulseRate.value = "2";
        elements.ikedaNoiseLevel.value = "0.5";
        elements.ikedaStripWidth.value = "0.05";
        elements.ikedaPhaseShift.value = "1.57";
        elements.ikedaQuantumLevels.value = "8";
        elements.fadeSlider.value = "0.5";
        elements.switchSlider.value = "0.33";
        elements.tileSlider.value = "3";
        elements.uploadsSlider.value = "0";
        elements.scrollSpeedX.value = "0.1";
        elements.scrollSpeedY.value = "0.00";
        elements.scrollOffsetX.value = "0.10";
        elements.scrollOffsetY.value = "0.00";
        
        // Trigger all change events
        const allControls = [
            'ikedaMode', 'ikedaThreshold', 'ikedaGridSize', 'ikedaDataIntensity',
            'ikedaFrequency', 'ikedaScanSpeed', 'ikedaMatrixScale', 'ikedaPulseRate',
            'ikedaNoiseLevel', 'ikedaStripWidth', 'ikedaPhaseShift', 'ikedaQuantumLevels',
            'fadeSlider', 'switchSlider', 'tileSlider', 'uploadsSlider',
            'scrollSpeedX', 'scrollSpeedY', 'scrollOffsetX', 'scrollOffsetY'
        ];
        
        allControls.forEach(controlName => {
            const element = elements[controlName];
            if (element) {
                const eventType = controlName === 'ikedaMode' ? 'change' : 'input';
                element.dispatchEvent(new Event(eventType));
            }
        });
        
        console.log("All controls reset to defaults");
    }

    function updateDataDisplay(analysisData) {
        if (analysisData) {
            elements.dataLuminance.textContent = (analysisData.luminance || 0).toFixed(3);
            elements.dataEntropy.textContent = (analysisData.entropy || 0).toFixed(3);
            elements.dataVariance.textContent = (analysisData.variance || 0).toFixed(3);
            elements.dataEdgeDensity.textContent = (analysisData.edge_density || 0).toFixed(3);
            elements.dataFreqRatio.textContent = (analysisData.freq_ratio || 0).toFixed(3);
            elements.dataCompression.textContent = (analysisData.compression || 0).toFixed(3);
        }
        
        // Update timestamp
        const now = new Date();
        elements.dataTimestamp.textContent = now.toLocaleTimeString();
    }

    function updateCounters() {
        // Update FPS counter
        const now = Date.now();
        if (now - lastFpsTime >= 1000) {
            fpsCounter = frameCount;
            frameCount = 0;
            lastFpsTime = now;
            elements.fpsCounter.textContent = fpsCounter;
        }
        
        // Update image counter
        elements.imageCounter.textContent = imageCounter;
    }

    function updateStatusBar(message, color) {
        elements.connectionStatus.textContent = message;
        elements.connectionStatus.style.color = color || "#FFFFFF";
    }

    function toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(err => {
                console.log(`Error attempting to enable fullscreen: ${err.message}`);
            });
        } else {
            document.exitFullscreen();
        }
    }

    // ------------------------------------------------------------------------
    // 8) Initialize Default State
    // ------------------------------------------------------------------------

    // Set initial mode and trigger UI update
    elements.ikedaMode.dispatchEvent(new Event('change'));
    
    // Initialize all control values
    const initControls = [
        'ikedaThreshold', 'ikedaGridSize', 'ikedaDataIntensity',
        'ikedaFrequency', 'ikedaScanSpeed', 'ikedaMatrixScale', 'ikedaPulseRate',
        'ikedaNoiseLevel', 'ikedaStripWidth', 'ikedaPhaseShift', 'ikedaQuantumLevels',
        'fadeSlider', 'switchSlider', 'tileSlider', 'uploadsSlider',
        'scrollSpeedX', 'scrollSpeedY', 'scrollOffsetX', 'scrollOffsetY'
    ];
    
    initControls.forEach(controlName => {
        const element = elements[controlName];
        if (element) {
            element.dispatchEvent(new Event('input'));
        }
    });

    console.log("Extended Ikeda control system initialized with 13 modes");
}; 