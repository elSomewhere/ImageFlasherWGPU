// Enhanced Ikeda-inspired control system for ImageFlasherWGPU
// Restructured with separate preprocessing and postprocessing pipeline

Module['onRuntimeInitialized'] = () => {
    console.log("WASM runtime initialized. Setting up restructured Ikeda control system...");

    function getImageWebSocketUrl() {
        const params = new URLSearchParams(window.location.search);
        const explicitUrl = params.get('imageWs');
        if (explicitUrl) return explicitUrl;

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const hostname = window.location.hostname || '127.0.0.1';
        return `${protocol}//${hostname}:5010`;
    }

    // ------------------------------------------------------------------------
    // 1) WebSocket Connection with Enhanced Data Handling
    // ------------------------------------------------------------------------
    const ws = new WebSocket(getImageWebSocketUrl());
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
    // 2) Enhanced C++ Function Wrappers - Restructured Pipeline
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

    // Restructured Pipeline Functions
    const setPreprocessingMode   = Module.cwrap('setPreprocessingMode', null, ['number']);
    const setPostprocessingMode  = Module.cwrap('setPostprocessingMode', null, ['number']);
    const setIkedaThreshold      = Module.cwrap('setIkedaThreshold', null, ['number']);
    const setIkedaGridSize       = Module.cwrap('setIkedaGridSize', null, ['number']);
    const setIkedaDataIntensity  = Module.cwrap('setIkedaDataIntensity', null, ['number']);
    
    // Color intensity maps to data intensity for color mode
    const setColorIntensity = setIkedaDataIntensity;
    
    // Note: Mode-specific parameters like frequency, scan speed, etc. are not yet 
    // implemented in the C++ backend. For now, they will use the core parameters.
    const setIkedaFrequency      = setIkedaDataIntensity; // Placeholder
    const setIkedaScanSpeed      = setIkedaDataIntensity; // Placeholder  
    const setIkedaMatrixScale    = setIkedaDataIntensity; // Placeholder
    const setIkedaPulseRate      = setIkedaDataIntensity; // Placeholder
    const setIkedaNoiseLevel     = setIkedaDataIntensity; // Placeholder
    const setIkedaStripWidth     = setIkedaDataIntensity; // Placeholder
    const setIkedaPhaseShift     = setIkedaDataIntensity; // Placeholder
    const setIkedaQuantumLevels  = setIkedaDataIntensity; // Placeholder

    // ------------------------------------------------------------------------
    // 3) Restructured UI Control System
    // ------------------------------------------------------------------------

    // Define mode names
    const preprocessingNames = ['COLOR', 'BLACK/WHITE'];
    const postprocessingNames = [
        'NONE', 'GRID', 'DATA', 'BINARY', 'FREQUENCY', 'SCAN', 
        'MATRIX', 'PULSE', 'NOISE', 'STRIP', 'PHASE', 'QUANTUM'
    ];

    // Preprocessing Mode Selection
    const preprocessingSelect = document.getElementById('preprocessingMode');
    preprocessingSelect.addEventListener('change', () => {
        const mode = parseInt(preprocessingSelect.value);
        setPreprocessingMode(mode);
        updatePreprocessingDisplay(preprocessingNames[mode]);
        showPreprocessingControls(mode);
        flashModeIndicator(preprocessingNames[mode] + ' PRE');
    });

    // Postprocessing Mode Selection
    const postprocessingSelect = document.getElementById('postprocessingMode');
    postprocessingSelect.addEventListener('change', () => {
        const mode = parseInt(postprocessingSelect.value);
        setPostprocessingMode(mode);
        updatePostprocessingDisplay(postprocessingNames[mode]);
        showModeControls(mode);
        flashModeIndicator(postprocessingNames[mode] + ' POST');
    });

    // Preprocessing Parameters
    setupSlider('ikedaThreshold', 'thresholdValue', setIkedaThreshold);
    setupSlider('colorIntensity', 'colorIntensityValue', setIkedaDataIntensity); // Now maps to dataIntensity

    // Mode-Specific Parameters (now includes relevant core processing controls)
    // Grid Mode
    setupSlider('gridSize', 'gridSizeValue', setIkedaGridSize);
    setupSlider('gridLines', 'gridLinesValue', (value) => setIkedaGridSize(value));
    
    // Data Mode
    setupSlider('dataIntensity', 'dataIntensityValue', setIkedaDataIntensity);
    setupSlider('dataOverlay', 'dataOverlayValue', (value) => setIkedaDataIntensity(value));
    
    // Binary Mode
    setupSlider('binaryCutoff', 'binaryCutoffValue', setIkedaThreshold);
    
    // Frequency Mode
    setupSlider('ikedaFrequency', 'frequencyValue', setIkedaFrequency);
    setupSlider('frequencyDataIntensity', 'frequencyDataIntensityValue', setIkedaDataIntensity);
    
    // Scan Mode
    setupSlider('ikedaScanSpeed', 'scanSpeedValue', setIkedaScanSpeed);
    setupSlider('scanGridSize', 'scanGridSizeValue', setIkedaGridSize);
    
    // Matrix Mode
    setupSlider('ikedaMatrixScale', 'matrixScaleValue', setIkedaMatrixScale);
    setupSlider('matrixGridSize', 'matrixGridSizeValue', setIkedaGridSize);
    
    // Pulse Mode
    setupSlider('ikedaPulseRate', 'pulseRateValue', setIkedaPulseRate);
    setupSlider('pulseDataIntensity', 'pulseDataIntensityValue', setIkedaDataIntensity);
    
    // Noise Mode
    setupSlider('ikedaNoiseLevel', 'noiseLevelValue', setIkedaNoiseLevel);
    setupSlider('noiseGridSize', 'noiseGridSizeValue', setIkedaGridSize);
    
    // Strip Mode
    setupSlider('ikedaStripWidth', 'stripWidthValue', setIkedaStripWidth);
    
    // Phase Mode
    setupSlider('ikedaPhaseShift', 'phaseShiftValue', setIkedaPhaseShift);
    setupSlider('phaseDataIntensity', 'phaseDataIntensityValue', setIkedaDataIntensity);
    
    // Quantum Mode
    setupSlider('ikedaQuantumLevels', 'quantumLevelsValue', setIkedaQuantumLevels);
    setupSlider('quantumGridSize', 'quantumGridSizeValue', setIkedaGridSize);

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
    // 4) Enhanced Keyboard Shortcuts - Restructured Controls
    // ------------------------------------------------------------------------
    document.addEventListener('keydown', (event) => {
        const key = event.key.toLowerCase();
        
        // Postprocessing mode selection shortcuts (1-9, 0)
        if (key >= '1' && key <= '9') {
            const mode = parseInt(key) - 1;
            if (mode < postprocessingNames.length) {
                selectPostprocessingMode(mode);
            }
            event.preventDefault();
        } else if (key === '0') {
            selectPostprocessingMode(10); // QUANTUM (index 10)
            event.preventDefault();
        }
        
        // Quick mode access
        switch (key) {
            case 'c': togglePreprocessing(); break; // Toggle Color/B&W
            case 'g': selectPostprocessingMode(1); break; // GRID
            case 'd': selectPostprocessingMode(2); break; // DATA
            case 'b': selectPostprocessingMode(3); break; // BINARY
            case 'f': selectPostprocessingMode(4); break; // FREQUENCY
            case 's': selectPostprocessingMode(5); break; // SCAN
            case 'm': selectPostprocessingMode(6); break; // MATRIX
            case 'p': selectPostprocessingMode(7); break; // PULSE
            case 'n': selectPostprocessingMode(8); break; // NOISE
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

    function selectPostprocessingMode(mode) {
        postprocessingSelect.value = mode;
        setPostprocessingMode(mode);
        updatePostprocessingDisplay(postprocessingNames[mode]);
        showModeControls(mode);
        flashModeIndicator(postprocessingNames[mode]);
    }

    function showModeControls(mode) {
        // Hide all mode controls
        document.querySelectorAll('.mode-controls').forEach(control => {
            control.classList.remove('active');
        });
        
        // Show relevant controls
        const controlMap = {
            0: 'noneControls',
            1: 'gridControls',
            2: 'dataControls',
            3: 'binaryControls',
            4: 'frequencyControls',
            5: 'scanControls', 
            6: 'matrixControls',
            7: 'pulseControls',
            8: 'noiseControls',
            9: 'stripControls',
            10: 'phaseControls',
            11: 'quantumControls'
        };
        
        const controlId = controlMap[mode];
        if (controlId) {
            const control = document.getElementById(controlId);
            if (control) control.classList.add('active');
        }
    }

    function showPreprocessingControls(mode) {
        // Hide color controls by default
        const colorControls = document.getElementById('colorControls');
        if (colorControls) {
            colorControls.classList.remove('active');
        }

        // Show color controls only in COLOR mode (mode 0)
        if (mode === 0 && colorControls) {
            colorControls.classList.add('active');
        }
        
        // Update threshold label based on mode
        updateThresholdLabel(mode);
    }

    function updateThresholdLabel(preprocessingMode) {
        const thresholdRow = document.getElementById('thresholdRow');
        if (thresholdRow) {
            const label = thresholdRow.querySelector('label');
            if (label) {
                if (preprocessingMode === 0) {
                    label.textContent = 'Contrast:';
                } else {
                    label.textContent = 'Threshold:';
                }
            }
        }
    }

    function togglePreprocessing() {
        const currentMode = parseInt(preprocessingSelect.value);
        const newMode = currentMode === 0 ? 1 : 0;
        preprocessingSelect.value = newMode;
        setPreprocessingMode(newMode);
        updatePreprocessingDisplay(preprocessingNames[newMode]);
        showPreprocessingControls(newMode);
        flashModeIndicator(preprocessingNames[newMode] + ' PRE');
    }

    function updatePreprocessingDisplay(modeName) {
        document.getElementById('currentPreprocessing').textContent = modeName;
    }

    function updatePostprocessingDisplay(modeName) {
        document.getElementById('currentPostprocessing').textContent = modeName;
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
        // Reset to default modes
        selectPostprocessingMode(2); // DATA mode (now at index 2)
        preprocessingSelect.value = 1; // BLACK/WHITE
        setPreprocessingMode(1);
        updatePreprocessingDisplay('BLACK/WHITE');
        showPreprocessingControls(1);
        
        // Reset preprocessing parameters
        setSliderValue('ikedaThreshold', 0.5);
        setSliderValue('colorIntensity', 1.0);
        
        // Reset mode-specific parameters to defaults
        setSliderValue('gridSize', 32);
        setSliderValue('gridLines', 16);
        setSliderValue('dataIntensity', 0.5);
        setSliderValue('dataOverlay', 0.7);
        setSliderValue('binaryCutoff', 0.5);
        setSliderValue('ikedaFrequency', 3.0);
        setSliderValue('frequencyDataIntensity', 0.5);
        setSliderValue('ikedaScanSpeed', 0.5);
        setSliderValue('scanGridSize', 32);
        setSliderValue('ikedaMatrixScale', 1.0);
        setSliderValue('matrixGridSize', 32);
        setSliderValue('ikedaPulseRate', 2.0);
        setSliderValue('pulseDataIntensity', 0.5);
        setSliderValue('ikedaNoiseLevel', 0.5);
        setSliderValue('noiseGridSize', 32);
        setSliderValue('ikedaStripWidth', 0.05);
        setSliderValue('ikedaPhaseShift', 1.57);
        setSliderValue('phaseDataIntensity', 0.5);
        setSliderValue('ikedaQuantumLevels', 8);
        setSliderValue('quantumGridSize', 32);
        
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

    setupCrawlerControls();

    // ------------------------------------------------------------------------
    // 7) Initialize System
    // ------------------------------------------------------------------------
    
    // Set initial modes
    selectPostprocessingMode(2); // DATA mode (now at index 2)
    preprocessingSelect.value = 1; // BLACK/WHITE
    setPreprocessingMode(1);
    updatePreprocessingDisplay('BLACK/WHITE');
    showPreprocessingControls(1);
    
    // Update status
    updateStatusBar("INITIALIZING...", "YELLOW");
    
    console.log("Restructured Ikeda control system initialized with preprocessing/postprocessing pipeline");
    console.log("Keyboard shortcuts active - Press C for color toggle, G/D/B/F/S/M/P/N for postprocessing modes");

    function parseListInput(value) {
        return value
            .split(',')
            .map((item) => item.trim())
            .filter(Boolean);
    }

    async function crawlerRequest(path, options = {}) {
        const response = await fetch(path, {
            headers: { 'Content-Type': 'application/json' },
            ...options
        });
        const body = await response.json();
        if (!response.ok || body.ok === false) {
            throw new Error(body.error || `Request failed: ${response.status}`);
        }
        return body;
    }

    function renderCrawlerState(state) {
        const payload = state.state || state;
        const keywords = (payload.keywords || []).join(', ') || 'none';
        const status = `keywords: ${keywords} | frontier: ${payload.frontier_size ?? '--'} | queue: ${payload.queue_size ?? '--'} | pages: ${payload.pages_visited ?? '--'} | candidates: ${payload.image_candidates ?? '--'} | images: ${payload.images_accepted ?? '--'}/${payload.images_rejected ?? '--'}`;
        const panel = document.getElementById('crawlerStatus');
        const label = document.getElementById('crawlerStatusLabel');
        if (panel) panel.textContent = status;
        if (label) label.textContent = `${payload.queue_size ?? 0}/${payload.images_accepted ?? 0}`;
        renderCrawlerLog(payload.recent_events || [], payload.recent_errors || []);
    }

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function renderCrawlerLog(events, errors) {
        const log = document.getElementById('crawlerLog');
        if (!log) return;

        const recentEvents = events.slice(-12).reverse();
        if (recentEvents.length === 0 && errors.length === 0) {
            log.textContent = 'Crawler log: no events yet';
            return;
        }

        const rows = recentEvents.map((event) => {
            const type = escapeHtml(event.type || 'event');
            const time = escapeHtml(event.time || '--:--:--');
            const message = escapeHtml(event.message || '');
            return `<div class="crawler-log-entry"><strong>${time} ${type}</strong>: ${message}</div>`;
        });

        if (errors.length > 0 && recentEvents.length === 0) {
            rows.push(...errors.slice(-5).reverse().map((error) => {
                return `<div class="crawler-log-entry"><strong>error</strong>: ${escapeHtml(error)}</div>`;
            }));
        }

        log.innerHTML = rows.join('');
    }

    async function refreshCrawlerState() {
        try {
            const state = await crawlerRequest('/api/crawler/state');
            renderCrawlerState(state);
        } catch (error) {
            const panel = document.getElementById('crawlerStatus');
            const label = document.getElementById('crawlerStatusLabel');
            const log = document.getElementById('crawlerLog');
            if (panel) panel.textContent = `Crawler: ${error.message}`;
            if (label) label.textContent = 'offline';
            if (log) log.textContent = `Crawler log: ${error.message}`;
        }
    }

    function setupCrawlerControls() {
        const keywordInput = document.getElementById('crawlerKeywords');
        const seedInput = document.getElementById('crawlerSeed');
        const keywordButton = document.getElementById('applyCrawlerKeywords');
        const seedButton = document.getElementById('addCrawlerSeed');

        if (keywordButton && keywordInput) {
            keywordButton.addEventListener('click', async () => {
                try {
                    const keywords = parseListInput(keywordInput.value);
                    const state = await crawlerRequest('/api/crawler/keywords', {
                        method: 'POST',
                        body: JSON.stringify({ keywords })
                    });
                    renderCrawlerState(state);
                    flashModeIndicator('CRAWLER KEYWORDS');
                } catch (error) {
                    document.getElementById('crawlerStatus').textContent = `Crawler: ${error.message}`;
                }
            });
        }

        if (seedButton && seedInput) {
            seedButton.addEventListener('click', async () => {
                try {
                    const seed = seedInput.value.trim();
                    const state = await crawlerRequest('/api/crawler/seeds', {
                        method: 'POST',
                        body: JSON.stringify({ seeds: seed ? [seed] : [] })
                    });
                    renderCrawlerState(state);
                    flashModeIndicator('CRAWLER SEED');
                } catch (error) {
                    document.getElementById('crawlerStatus').textContent = `Crawler: ${error.message}`;
                }
            });
        }

        refreshCrawlerState();
        setInterval(refreshCrawlerState, 3000);
    }
};

// Enhanced error handling for WebAssembly initialization
Module['onAbort'] = (what) => {
    console.error("WebAssembly module aborted:", what);
    document.getElementById('connectionStatus').textContent = "WASM ERROR";
};

console.log("Restructured Ikeda control system loading..."); 