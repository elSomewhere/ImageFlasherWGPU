// Enhanced Image Processing Control System for ImageFlasherWGPU
// Implements genuine image manipulation modes with proper parameter controls

Module['onRuntimeInitialized'] = () => {
    console.log("WASM runtime initialized. Setting up enhanced image processing control system...");

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
        console.log("WebSocket connected - Enhanced image processing active");
        updateStatusBar("CONNECTED", "WHITE");
    };

    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        updateStatusBar("CONNECTION ERROR", "RED");
    };

    ws.onmessage = (event) => {
        let byteArray = new Uint8Array(event.data);
        
        // Check if this is a metadata-packed message
        if (byteArray.length > 4) {
            try {
                const metadataSize = new DataView(byteArray.buffer, 0, 4).getUint32(0, true);
                
                if (metadataSize > 0 && metadataSize < byteArray.length) {
                    const metadataBytes = byteArray.slice(4, 4 + metadataSize);
                    const metadataStr = new TextDecoder().decode(metadataBytes);
                    const metadata = JSON.parse(metadataStr);
                    const imageData = byteArray.slice(4 + metadataSize);
                    
                    updateDataDisplay(metadata.analysis);
                    
                    let ptr = Module._malloc(imageData.length);
                    Module.HEAPU8.set(imageData, ptr);
                    Module.ccall('onImageReceived', null, ['number', 'number'], [ptr, imageData.length]);
                    Module._free(ptr);
                } else {
                    let ptr = Module._malloc(byteArray.length);
                    Module.HEAPU8.set(byteArray, ptr);
                    Module.ccall('onImageReceived', null, ['number', 'number'], [ptr, byteArray.length]);
                    Module._free(ptr);
                }
            } catch (e) {
                let ptr = Module._malloc(byteArray.length);
                Module.HEAPU8.set(byteArray, ptr);
                Module.ccall('onImageReceived', null, ['number', 'number'], [ptr, byteArray.length]);
                Module._free(ptr);
            }
        } else {
            let ptr = Module._malloc(byteArray.length);
            Module.HEAPU8.set(byteArray, ptr);
            Module.ccall('onImageReceived', null, ['number', 'number'], [ptr, byteArray.length]);
            Module._free(ptr);
        }

        imageCounter++;
        frameCount++;
        document.getElementById('imageCounter').textContent = imageCounter;
        
        const now = Date.now();
        if (now - lastFpsTime >= 1000) {
            fpsCounter = frameCount;
            frameCount = 0;
            lastFpsTime = now;
            document.getElementById('fpsCounter').textContent = fpsCounter;
        }
    };

    // ------------------------------------------------------------------------
    // 2) Enhanced C++ Function Wrappers
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

    // Core processing functions
    const setPreprocessingMode   = Module.cwrap('setPreprocessingMode', null, ['number']);
    const setPostprocessingMode  = Module.cwrap('setPostprocessingMode', null, ['number']);
    const setIkedaThreshold      = Module.cwrap('setIkedaThreshold', null, ['number']);
    const setIkedaGridSize       = Module.cwrap('setIkedaGridSize', null, ['number']);
    const setIkedaDataIntensity  = Module.cwrap('setIkedaDataIntensity', null, ['number']);
    
    // Enhanced parameter functions (with fallbacks for compatibility)
    const setContrastBoost       = Module.cwrap('setContrastBoost', null, ['number']) || setIkedaDataIntensity;
    const setAdaptiveRadius      = Module.cwrap('setAdaptiveRadius', null, ['number']) || setIkedaDataIntensity;
    const setThresholdSmoothing  = Module.cwrap('setThresholdSmoothing', null, ['number']) || setIkedaDataIntensity;
    const setGridRotation        = Module.cwrap('setGridRotation', null, ['number']) || setIkedaDataIntensity;
    const setGridAspect          = Module.cwrap('setGridAspect', null, ['number']) || setIkedaDataIntensity;
    const setEdgeSensitivity     = Module.cwrap('setEdgeSensitivity', null, ['number']) || setIkedaDataIntensity;
    const setChannelOffsetX      = Module.cwrap('setChannelOffsetX', null, ['number']) || setIkedaDataIntensity;
    const setChannelOffsetY      = Module.cwrap('setChannelOffsetY', null, ['number']) || setIkedaDataIntensity;
    const setChannelMixAmount    = Module.cwrap('setChannelMixAmount', null, ['number']) || setIkedaDataIntensity;
    const setPixelPattern        = Module.cwrap('setPixelPattern', null, ['number']) || setIkedaDataIntensity;
    const setDisplacementStrength = Module.cwrap('setDisplacementStrength', null, ['number']) || setIkedaDataIntensity;
    const setStripOrientation    = Module.cwrap('setStripOrientation', null, ['number']) || setIkedaDataIntensity;
    const setStripBlendMode      = Module.cwrap('setStripBlendMode', null, ['number']) || setIkedaDataIntensity;
    const setStripDistortion     = Module.cwrap('setStripDistortion', null, ['number']) || setIkedaDataIntensity;
    const setQuantumCurve        = Module.cwrap('setQuantumCurve', null, ['number']) || setIkedaDataIntensity;
    const setDithering           = Module.cwrap('setDithering', null, ['number']) || setIkedaDataIntensity;

    // ------------------------------------------------------------------------
    // 3) Enhanced UI Control System
    // ------------------------------------------------------------------------

    // Define mode names
    const preprocessingNames = ['COLOR', 'BLACK/WHITE'];
    const postprocessingNames = [
        'NONE', 'GRID', 'EDGES', 'BINARY', 'CHANNELS', 'PIXELATION', 
        'DISPLACEMENT', 'STRIPS', 'QUANTUM'
    ];

    let currentPreprocessingMode = 1; // Default to BLACK/WHITE
    let currentPostprocessingMode = 2; // Default to EDGES

    // Preprocessing Mode Selection
    const preprocessingSelect = document.getElementById('preprocessingMode');
    preprocessingSelect.addEventListener('change', () => {
        currentPreprocessingMode = parseInt(preprocessingSelect.value);
        setPreprocessingMode(currentPreprocessingMode);
        updatePreprocessingDisplay(preprocessingNames[currentPreprocessingMode]);
        showPreprocessingControls(currentPreprocessingMode);
        flashModeIndicator(preprocessingNames[currentPreprocessingMode] + ' PRE');
    });

    // Postprocessing Mode Selection
    const postprocessingSelect = document.getElementById('postprocessingMode');
    postprocessingSelect.addEventListener('change', () => {
        currentPostprocessingMode = parseInt(postprocessingSelect.value);
        setPostprocessingMode(currentPostprocessingMode);
        updatePostprocessingDisplay(postprocessingNames[currentPostprocessingMode]);
        showModeControls(currentPostprocessingMode);
        flashModeIndicator(postprocessingNames[currentPostprocessingMode]);
    });

    // Setup all sliders with enhanced functionality
    function setupSlider(sliderId, valueId, setterFunction, formatter = null) {
        const slider = document.getElementById(sliderId);
        const valueDisplay = document.getElementById(valueId);
        
        if (!slider || !valueDisplay) {
            console.warn(`Slider ${sliderId} or value display ${valueId} not found`);
            return;
        }

        slider.addEventListener('input', () => {
            const value = parseFloat(slider.value);
            setterFunction(value);
            
            if (formatter) {
                valueDisplay.textContent = formatter(value);
            } else {
                valueDisplay.textContent = value.toFixed(2);
            }
        });

        // Initialize display
        const initialValue = parseFloat(slider.value);
        if (formatter) {
            valueDisplay.textContent = formatter(initialValue);
        } else {
            valueDisplay.textContent = initialValue.toFixed(2);
        }
        setterFunction(initialValue);
    }

    // Enhanced formatters for special values
    const formatters = {
        pixelPattern: (value) => {
            const patterns = ['Square', 'Hexagon', 'Triangle', 'Voronoi'];
            return patterns[Math.floor(value * 4)] || 'Square';
        },
        stripBlendMode: (value) => {
            const modes = ['Normal', 'Multiply', 'Screen'];
            return modes[Math.floor(value * 3)] || 'Normal';
        }
    };

    // Core parameters
    setupSlider('ikedaThreshold', 'thresholdValue', setIkedaThreshold);
    setupSlider('dataIntensity', 'dataIntensityValue', setIkedaDataIntensity);
    setupSlider('colorIntensity', 'colorIntensityValue', setIkedaDataIntensity);

    // Enhanced Black/White parameters
    setupSlider('contrastBoost', 'contrastBoostValue', setContrastBoost);
    setupSlider('adaptiveRadius', 'adaptiveRadiusValue', setAdaptiveRadius);
    setupSlider('thresholdSmoothing', 'thresholdSmoothingValue', setThresholdSmoothing);

    // Grid parameters
    setupSlider('gridSize', 'gridSizeValue', setIkedaGridSize);
    setupSlider('gridRotation', 'gridRotationValue', setGridRotation);
    setupSlider('gridAspect', 'gridAspectValue', setGridAspect);

    // Edge detection parameters
    setupSlider('edgeSensitivity', 'edgeSensitivityValue', setEdgeSensitivity);

    // Binary parameters
    setupSlider('binaryScanlines', 'binaryScanlinesValue', setIkedaGridSize);

    // Channel manipulation parameters
    setupSlider('channelOffsetX', 'channelOffsetXValue', setChannelOffsetX);
    setupSlider('channelOffsetY', 'channelOffsetYValue', setChannelOffsetY);
    setupSlider('channelMixAmount', 'channelMixAmountValue', setChannelMixAmount);

    // Pixelation parameters
    setupSlider('pixelationSize', 'pixelationSizeValue', setIkedaGridSize);
    setupSlider('pixelPattern', 'pixelPatternValue', setPixelPattern, formatters.pixelPattern);

    // Displacement parameters
    setupSlider('displacementStrength', 'displacementStrengthValue', setDisplacementStrength);

    // Strip parameters
    setupSlider('stripOrientation', 'stripOrientationValue', setStripOrientation);
    setupSlider('stripBlendMode', 'stripBlendModeValue', setStripBlendMode, formatters.stripBlendMode);
    setupSlider('stripDistortion', 'stripDistortionValue', setStripDistortion);

    // Quantum parameters
    setupSlider('quantumLevels', 'quantumLevelsValue', setIkedaGridSize);
    setupSlider('quantumCurve', 'quantumCurveValue', setQuantumCurve);
    setupSlider('dithering', 'ditheringValue', setDithering);

    // Motion and performance controls
    setupSlider('scrollSpeedX', 'scrollSpeedXValue', setScrollSpeedX);
    setupSlider('scrollSpeedY', 'scrollSpeedYValue', setScrollSpeedY);
    setupSlider('imageSwitchInterval', 'imageSwitchIntervalValue', setImageSwitchInterval);
    setupSlider('tileFactor', 'tileFactorValue', setTileFactor);
    setupSlider('fadeFactor', 'fadeFactorValue', setFadeFactor);

    // ------------------------------------------------------------------------
    // 4) Mode Control Functions
    // ------------------------------------------------------------------------

    function showPreprocessingControls(mode) {
        // Hide all preprocessing controls
        document.getElementById('colorControls').classList.remove('active');
        document.getElementById('blackWhiteControls').classList.remove('active');
        
        if (mode === 0) {
            document.getElementById('colorControls').classList.add('active');
        } else if (mode === 1) {
            document.getElementById('blackWhiteControls').classList.add('active');
        }
    }

    function showModeControls(mode) {
        // Hide all mode controls
        const modeControls = [
            'gridControls', 'edgeControls', 'binaryControls', 'channelControls',
            'pixelationControls', 'displacementControls', 'stripControls', 'quantumControls'
        ];
        
        modeControls.forEach(id => {
            const element = document.getElementById(id);
            if (element) element.classList.remove('active');
        });
        
        // Show controls for current mode
        const controlMap = {
            1: 'gridControls',
            2: 'edgeControls',
            3: 'binaryControls',
            4: 'channelControls',
            5: 'pixelationControls',
            6: 'displacementControls',
            7: 'stripControls',
            8: 'quantumControls'
        };
        
        const controlId = controlMap[mode];
        if (controlId) {
            const element = document.getElementById(controlId);
            if (element) element.classList.add('active');
        }
    }

    function updatePreprocessingDisplay(name) {
        document.getElementById('processingMode').textContent = name;
    }

    function updatePostprocessingDisplay(name) {
        document.getElementById('currentMode').textContent = name;
    }

    function flashModeIndicator(text) {
        const indicator = document.getElementById('modeIndicator');
        indicator.textContent = text;
        indicator.classList.remove('mode-flash');
        setTimeout(() => indicator.classList.add('mode-flash'), 10);
    }

    function updateStatusBar(status, color) {
        const statusElement = document.getElementById('connectionStatus');
        statusElement.textContent = status;
        statusElement.style.color = color || '#FFFFFF';
    }

    // ------------------------------------------------------------------------
    // 5) Data Display Updates
    // ------------------------------------------------------------------------

    function updateDataDisplay(analysis) {
        if (!analysis) return;
        
        const elements = {
            luminanceValue: analysis.averageLuminance?.toFixed(3) || '0.000',
            entropyValue: analysis.entropy?.toFixed(3) || '0.000',
            varianceValue: analysis.variance?.toFixed(3) || '0.000',
            edgeDensityValue: analysis.edgeDensity?.toFixed(3) || '0.000'
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) element.textContent = value;
        });
    }

    // ------------------------------------------------------------------------
    // 6) Keyboard Shortcuts
    // ------------------------------------------------------------------------

    document.addEventListener('keydown', (event) => {
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') {
            return; // Don't handle shortcuts when typing in inputs
        }

        const key = event.key.toLowerCase();
        event.preventDefault();

        switch (key) {
            case '1': setModeFromKeyboard(1); break;
            case '2': setModeFromKeyboard(2); break;
            case '3': setModeFromKeyboard(3); break;
            case '4': setModeFromKeyboard(4); break;
            case '5': setModeFromKeyboard(5); break;
            case '6': setModeFromKeyboard(6); break;
            case '7': setModeFromKeyboard(7); break;
            case '8': setModeFromKeyboard(8); break;
            case '0': setModeFromKeyboard(0); break;
            
            case 'g': setModeFromKeyboard(1); break; // Grid
            case 'e': setModeFromKeyboard(2); break; // Edges
            case 'b': setModeFromKeyboard(3); break; // Binary
            case 'c': setModeFromKeyboard(4); break; // Channels
            case 'p': setModeFromKeyboard(5); break; // Pixelation
            case 'd': setModeFromKeyboard(6); break; // Displacement
            case 's': setModeFromKeyboard(7); break; // Strips
            case 'q': setModeFromKeyboard(8); break; // Quantum
            
            case 't': cycleThreshold(); break;
            case 'r': resetToDefaults(); break;
            case 'escape': toggleControls(); break;
        }
    });

    function setModeFromKeyboard(mode) {
        postprocessingSelect.value = mode;
        postprocessingSelect.dispatchEvent(new Event('change'));
    }

    function cycleThreshold() {
        const thresholds = [0.3, 0.5, 0.7];
        const currentThreshold = parseFloat(document.getElementById('ikedaThreshold').value);
        let nextIndex = thresholds.findIndex(t => t > currentThreshold);
        if (nextIndex === -1) nextIndex = 0;
        
        const slider = document.getElementById('ikedaThreshold');
        slider.value = thresholds[nextIndex];
        slider.dispatchEvent(new Event('input'));
    }

    function resetToDefaults() {
        // Reset all sliders to their default values
        const defaults = {
            'preprocessingMode': 1,
            'postprocessingMode': 2,
            'ikedaThreshold': 0.5,
            'dataIntensity': 0.7,
            'contrastBoost': 1.5,
            'adaptiveRadius': 5,
            'thresholdSmoothing': 0.02,
            'gridSize': 32,
            'gridRotation': 0,
            'gridAspect': 1.0,
            'edgeSensitivity': 2.0,
            'scrollSpeedX': 0.5,
            'scrollSpeedY': 0.3,
            'tileFactor': 4,
            'fadeFactor': 0.85
        };
        
        Object.entries(defaults).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.value = value;
                element.dispatchEvent(new Event('change'));
                element.dispatchEvent(new Event('input'));
            }
        });
        
        flashModeIndicator('RESET TO DEFAULTS');
    }

    function toggleControls() {
        const controls = document.getElementById('controls');
        const shortcuts = document.getElementById('shortcuts');
        const dataDisplay = document.getElementById('dataDisplay');
        
        const isVisible = controls.style.display !== 'none';
        const newDisplay = isVisible ? 'none' : 'block';
        
        controls.style.display = newDisplay;
        shortcuts.style.display = newDisplay;
        dataDisplay.style.display = newDisplay;
    }

    // ------------------------------------------------------------------------
    // 7) Reset Button Handler
    // ------------------------------------------------------------------------

    document.getElementById('resetButton').addEventListener('click', resetToDefaults);

    // ------------------------------------------------------------------------
    // 8) Buffer Usage Updates
    // ------------------------------------------------------------------------

    setInterval(() => {
        try {
            const usage = getBufferUsage();
            document.getElementById('bufferUsage').textContent = `${usage.toFixed(1)}%`;
        } catch (e) {
            // Buffer usage function not available
        }
    }, 1000);

    // ------------------------------------------------------------------------
    // 9) Initialize Interface
    // ------------------------------------------------------------------------

    // Set initial modes
    showPreprocessingControls(currentPreprocessingMode);
    showModeControls(currentPostprocessingMode);
    updatePreprocessingDisplay(preprocessingNames[currentPreprocessingMode]);
    updatePostprocessingDisplay(postprocessingNames[currentPostprocessingMode]);
    
    console.log("Enhanced image processing control system initialized successfully");
};

// Enhanced error handling for WebAssembly initialization
Module['onAbort'] = (what) => {
    console.error("WebAssembly module aborted:", what);
    document.getElementById('connectionStatus').textContent = "WASM ERROR";
};

console.log("Enhanced image processing control system loading..."); 