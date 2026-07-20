// DATAVALANCHE control layer
//
// - artifact stream: unchanged wire protocol (length-prefixed JSON header + payload)
// - engine: cwrap bindings for the RenderParams API in main.cpp
// - AudioEngine: data sonification — clicks, granular texture from raw artifact
//   bytes, analysis-pitched sine grid, subs/kicks, crawl-telemetry sonics
// - Conductor: autonomous scene evolution, reveals, dropouts, beat-locked glitches
// - Debug mode: neutral bypass showing the images exactly as collected
// - panel: minimal hidden control surface + crawler steering

Module['onRuntimeInitialized'] = () => {
    console.log('WASM runtime initialized: DATAVALANCHE presentation layer');

    // ------------------------------------------------------------------
    // Runtime config / WebSocket URL
    // ------------------------------------------------------------------
    let runtimeConfigPromise = null;

    function getRuntimeConfig() {
        if (!runtimeConfigPromise) {
            runtimeConfigPromise = fetch('/api/runtime-config')
                .then((response) => {
                    if (!response.ok) throw new Error(`Runtime config: ${response.status}`);
                    return response.json();
                })
                .catch((error) => {
                    console.warn('Runtime configuration unavailable; using defaults', error);
                    return { websocket_port: 5010, crawler_enabled: true };
                });
        }
        return runtimeConfigPromise;
    }

    async function getImageWebSocketUrl() {
        const params = new URLSearchParams(window.location.search);
        const explicitUrl = params.get('imageWs');
        if (explicitUrl) return explicitUrl;
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const hostname = window.location.hostname || '127.0.0.1';
        const config = await getRuntimeConfig();
        const port = Number(config.websocket_port) || 5010;
        return `${protocol}//${hostname}:${port}`;
    }

    // ------------------------------------------------------------------
    // Engine bindings
    // ------------------------------------------------------------------
    const engine = {
        setStyles:            Module.cwrap('setStyles', null, ['number', 'number', 'number']),
        setTone:              Module.cwrap('setTone', null, ['number', 'number', 'number', 'number']),
        setStructure:         Module.cwrap('setStructure', null, ['number', 'number', 'number', 'number']),
        setMosh:              Module.cwrap('setMosh', null, ['number', 'number', 'number', 'number']),
        setTemporal:          Module.cwrap('setTemporal', null, ['number', 'number', 'number', 'number']),
        setAccents:           Module.cwrap('setAccents', null, ['number', 'number']),
        setBypass:            Module.cwrap('setBypass', null, ['number']),
        setSequence:          Module.cwrap('setSequence', null, ['number']),
        pulse:                Module.cwrap('pulse', null, ['number']),
        setFadeFactor:        Module.cwrap('setFadeFactor', null, ['number']),
        setImageSwitchInterval: Module.cwrap('setImageSwitchInterval', null, ['number']),
        setTileFactor:        Module.cwrap('setTileFactor', null, ['number']),
        setRandomTileFraction: Module.cwrap('setRandomTileFraction', null, ['number']),
        setScrollingSpeed:    Module.cwrap('setScrollingSpeed', null, ['number', 'number']),
        setMaxUploadsPerFrame: Module.cwrap('setMaxUploadsPerFrame', null, ['number']),
        getBufferUsage:       Module.cwrap('getBufferUsage', 'number', []),
        getRingBufferSize:    Module.cwrap('getRingBufferSize', 'number', [])
    };

    const STYLE_NAMES = [
        'RAW', 'THRESH', 'BAYER', 'EDGE', 'SORT', 'SLICE',
        'WAVE', 'BARCODE', 'HEX', 'BLOCKS', 'BITPLANE', 'CONTOUR'
    ];

    // Central parameter state; every push reads from here.
    const state = {
        styleA: 2, styleB: 8, styleMix: 0.15,
        threshold: 0.5, contrast: 1.4, jitter: 0.3, colorBleed: 0,
        dither: 3, blockScale: 14, sliceAmp: 0.15, grid: 0.5,
        moshAmount: 0.25, moshBlock: 0.035, moshDrop: 0.15, moshDecay: 0.97,
        scanline: 0.35, noise: 0.08, strobe: 0, invert: 0,
        fade: 0.5, switchInterval: 0.33, tileFactor: 3, tileFraction: 0.5,
        scrollX: 0.06, scrollY: 0, uploads: 0,
        flashBoost: 0.6, gutter: 0
    };

    // Drift offsets applied on top of state by the conductor (visual breathing).
    const drift = { threshold: 0, moshAmount: 0, sliceAmp: 0 };

    function clamp01(x) { return Math.min(1, Math.max(0, x)); }

    function pushStyles()    { engine.setStyles(state.styleA, state.styleB, state.styleMix); }
    function pushTone()      { engine.setTone(clamp01(state.threshold + drift.threshold), state.contrast, state.colorBleed, state.jitter); }
    function pushStructure() { engine.setStructure(state.dither, state.blockScale, clamp01(state.sliceAmp + drift.sliceAmp), state.grid); }
    function pushMosh()      { engine.setMosh(clamp01(state.moshAmount + drift.moshAmount), state.moshBlock, state.moshDrop, state.moshDecay); }
    function pushTemporal()  { engine.setTemporal(state.scanline, state.noise, state.strobe, state.invert); }
    function pushAccents()   { engine.setAccents(state.flashBoost, state.gutter); }
    function pushFlow() {
        engine.setFadeFactor(state.fade);
        engine.setImageSwitchInterval(state.switchInterval);
        engine.setTileFactor(state.tileFactor);
        engine.setRandomTileFraction(state.tileFraction);
        engine.setScrollingSpeed(state.scrollX, state.scrollY);
        engine.setMaxUploadsPerFrame(state.uploads);
    }
    function pushAll() { pushStyles(); pushTone(); pushStructure(); pushMosh(); pushTemporal(); pushAccents(); pushFlow(); }

    // ------------------------------------------------------------------
    // AudioEngine: the crawl made audible
    // ------------------------------------------------------------------
    const DEFAULT_PROFILE = {
        bpm: 128,
        pitchSet: [1244.5, 1661.2, 2217.5, 2960.0, 3951.1, 5274.0, 7040.0],
        gridProb: 0.7,     // sine-grid blip probability per 16th
        clickProb: 0.2,    // extra click probability per 16th
        grainProb: 0.5,    // data-grain probability per 16th
        subPattern: false, // 8th-note sub pulse
        grainPitch: 1.0,   // playback-rate multiplier for grains
        bed: 0.02,         // noise-bed level
        sweep: 2400        // noise-bed bandpass center
    };

    class AudioEngine {
        constructor() {
            this.ctx = null;
            this.enabled = false;
            this.level = 0.6;
            this.density = 0.5;
            this.profile = { ...DEFAULT_PROFILE };
            this.analysis = null;
            this.nextNoteTime = 0;
            this.gridStep = 0;
            this.lastGrainAdd = 0;
            this.lastClick = 0;
            this.grainPool = [];      // AudioBuffers built from raw artifact bytes
            this.gridProbMod = 1.0;   // modulated by crawl accept/reject ratio
            this.hostCount = null;
        }

        async start() {
            if (this.ctx) {
                await this.ctx.resume();
                this.enabled = true;
                return;
            }
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            this.ctx = ctx;

            this.master = ctx.createGain();
            this.master.gain.value = this.level;
            this.limiter = ctx.createDynamicsCompressor();
            this.limiter.threshold.value = -18;
            this.limiter.knee.value = 6;
            this.limiter.ratio.value = 20;
            this.limiter.attack.value = 0.002;
            this.limiter.release.value = 0.12;
            this.master.connect(this.limiter);
            this.limiter.connect(ctx.destination);

            // impulse buffer for clicks: 64 samples of alternating polarity
            const impulse = ctx.createBuffer(1, 64, ctx.sampleRate);
            const imp = impulse.getChannelData(0);
            for (let i = 0; i < 64; i++) imp[i] = (i % 2 === 0 ? 1 : -1) * Math.exp(-i / 12);
            this.impulseBuf = impulse;

            // continuous filtered-noise bed, level driven by crawl frontier size
            const noiseBuf = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
            const nd = noiseBuf.getChannelData(0);
            for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
            this.noiseSrc = ctx.createBufferSource();
            this.noiseSrc.buffer = noiseBuf;
            this.noiseSrc.loop = true;
            this.noiseFilter = ctx.createBiquadFilter();
            this.noiseFilter.type = 'bandpass';
            this.noiseFilter.frequency.value = this.profile.sweep;
            this.noiseFilter.Q.value = 14;
            this.noiseGain = ctx.createGain();
            this.noiseGain.gain.value = 0.0;
            this.noiseSrc.connect(this.noiseFilter);
            this.noiseFilter.connect(this.noiseGain);
            this.noiseGain.connect(this.master);
            this.noiseSrc.start();

            this.nextNoteTime = ctx.currentTime + 0.1;
            this.timer = setInterval(() => this.schedule(), 90);
            this.enabled = true;
        }

        async stop() {
            if (!this.ctx) return;
            this.enabled = false;
            await this.ctx.suspend();
        }

        setLevel(v) {
            this.level = v;
            if (this.master) this.master.gain.setTargetAtTime(v, this.ctx.currentTime, 0.05);
        }
        setDensity(v) { this.density = v; }

        // panned output node for one voice
        out(pan) {
            const p = this.ctx.createStereoPanner();
            p.pan.value = Math.max(-1, Math.min(1, pan || 0));
            p.connect(this.master);
            return p;
        }

        click(t, peak = 0.4, pan = 0) {
            if (!this.enabled) return;
            const src = this.ctx.createBufferSource();
            src.buffer = this.impulseBuf;
            const g = this.ctx.createGain();
            g.gain.value = peak;
            src.connect(g);
            g.connect(this.out(pan));
            src.start(t);
        }

        clickBurst(n, spacing = 0.024, peak = 0.4) {
            if (!this.enabled) return;
            const t0 = this.ctx.currentTime;
            for (let i = 0; i < n; i++) {
                this.click(t0 + i * spacing, peak * (1 - i / (n + 2)), Math.random() * 1.4 - 0.7);
            }
        }

        blip(t, freq, dur = 0.05, peak = 0.16, pan = 0) {
            if (!this.enabled) return;
            const osc = this.ctx.createOscillator();
            osc.type = 'sine';
            osc.frequency.value = freq;
            const g = this.ctx.createGain();
            g.gain.setValueAtTime(0, t);
            g.gain.linearRampToValueAtTime(peak, t + 0.002);
            g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
            osc.connect(g);
            g.connect(this.out(pan));
            osc.start(t);
            osc.stop(t + dur + 0.02);
        }

        sub(t, freq = 45, dur = 0.5, peak = 0.5) {
            if (!this.enabled) return;
            const osc = this.ctx.createOscillator();
            osc.type = 'sine';
            osc.frequency.value = freq;
            const g = this.ctx.createGain();
            g.gain.setValueAtTime(0, t);
            g.gain.linearRampToValueAtTime(peak, t + 0.01);
            g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
            osc.connect(g);
            g.connect(this.master);
            osc.start(t);
            osc.stop(t + dur + 0.05);
        }

        // pitch-dropping sine kick on scene cuts
        kick(t) {
            if (!this.enabled) return;
            const osc = this.ctx.createOscillator();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(66, t);
            osc.frequency.exponentialRampToValueAtTime(38, t + 0.12);
            const g = this.ctx.createGain();
            g.gain.setValueAtTime(0, t);
            g.gain.linearRampToValueAtTime(0.7, t + 0.005);
            g.gain.exponentialRampToValueAtTime(0.0001, t + 0.35);
            osc.connect(g);
            g.connect(this.master);
            osc.start(t);
            osc.stop(t + 0.4);
        }

        // Build an AudioBuffer from raw artifact bytes and keep it in the pool.
        addGrainSource(bytes) {
            if (!this.ctx) return;
            const nowMs = performance.now();
            if (nowMs - this.lastGrainAdd < 400) return;
            this.lastGrainAdd = nowMs;
            const ctx = this.ctx;
            const n = Math.min(bytes.length, Math.floor(ctx.sampleRate * 0.35));
            if (n < 512) return;
            const stride = Math.max(1, Math.floor(bytes.length / n));
            const buf = ctx.createBuffer(1, n, ctx.sampleRate);
            const d = buf.getChannelData(0);
            for (let i = 0; i < n; i++) d[i] = (bytes[i * stride] - 128) / 128;
            this.grainPool.push(buf);
            if (this.grainPool.length > 8) this.grainPool.shift();
        }

        // One short windowed grain of collected data.
        grain(t) {
            if (!this.enabled || this.grainPool.length === 0) return;
            const ctx = this.ctx;
            const buf = this.grainPool[Math.floor(Math.random() * this.grainPool.length)];
            const dur = 0.01 + Math.random() * 0.07;
            const maxOffset = Math.max(0, buf.duration - dur - 0.001);
            const src = ctx.createBufferSource();
            src.buffer = buf;
            const rates = [0.5, 0.75, 1, 1.5, 2];
            src.playbackRate.value = rates[Math.floor(Math.random() * rates.length)] * this.profile.grainPitch;
            const filter = ctx.createBiquadFilter();
            filter.type = 'bandpass';
            filter.frequency.value = 300 + Math.random() * 5700;
            filter.Q.value = 1.2;
            const g = ctx.createGain();
            const peak = (0.05 + 0.13 * this.density);
            g.gain.setValueAtTime(0, t);
            g.gain.linearRampToValueAtTime(peak, t + dur * 0.5);   // hann-ish window
            g.gain.linearRampToValueAtTime(0, t + dur);
            src.connect(filter);
            filter.connect(g);
            g.connect(this.out(Math.random() * 1.6 - 0.8));
            src.start(t, Math.random() * maxOffset, dur + 0.01);
        }

        // 16th-note lookahead scheduler.
        schedule() {
            if (!this.enabled) return;
            const prof = this.profile;
            const spb = 60 / prof.bpm / 4;
            while (this.nextNoteTime < this.ctx.currentTime + 0.22) {
                const t = this.nextNoteTime;
                const d = this.density;
                if (Math.random() < prof.gridProb * this.gridProbMod * d * 1.2) {
                    const lum = this.analysis && this.analysis.luminance != null
                        ? clamp01(this.analysis.luminance) : Math.random();
                    const set = prof.pitchSet;
                    const idx = Math.min(set.length - 1, Math.floor(lum * set.length));
                    const entropy = this.analysis && this.analysis.entropy ? this.analysis.entropy / 8 : 0.5;
                    this.blip(t, set[idx], 0.03 + 0.09 * entropy, 0.16, (Math.random() - 0.5) * 0.6);
                }
                if (Math.random() < prof.clickProb * d) this.click(t, 0.3, Math.random() * 1.2 - 0.6);
                if (Math.random() < prof.grainProb * d) this.grain(t);
                if (prof.subPattern && this.gridStep % 8 === 0) this.sub(t, 45, 0.22, 0.45);
                if (this.gridStep % 16 === 0 && Math.random() < 0.35) this.sub(t, 41 + Math.random() * 8, 0.4, 0.4);
                this.gridStep++;
                this.nextNoteTime += spb;
            }
        }

        onArtifact(header, payload) {
            const analysis = header.metadata && header.metadata.analysis;
            if (analysis) this.analysis = analysis;
            if (!this.enabled) return;
            const nowMs = performance.now();
            if (nowMs - this.lastClick > 40) {
                this.lastClick = nowMs;
                // pan deterministically by sequence so each artifact has a place
                const h = ((header.sequence * 2654435761) >>> 0) / 4294967296;
                this.click(this.ctx.currentTime, 0.35, h * 1.6 - 0.8);
            }
            this.addGrainSource(payload);
        }

        onScene(scene) {
            this.profile = { ...DEFAULT_PROFILE, ...(scene.audio || {}) };
            if (!this.enabled) return;
            const t = this.ctx.currentTime;
            this.kick(t);
            this.clickBurst(9, 0.02, 0.5);
            this.noiseFilter.frequency.setTargetAtTime(this.profile.sweep, t, 0.4);
            this.noiseGain.gain.setTargetAtTime(this.profile.bed * this.density, t, 0.5);
        }

        onPulse(strength) {
            if (!this.enabled) return;
            this.clickBurst(3 + Math.floor(strength * 6), 0.018, 0.45);
        }

        // Crawl telemetry -> sound: new hosts ping, frontier feeds the bed,
        // rejection ratio thins the grid.
        onTelemetry(payload) {
            if (payload.distinct_hosts != null) {
                if (this.hostCount != null && payload.distinct_hosts > this.hostCount && this.enabled) {
                    const t = this.ctx.currentTime;
                    [5274, 6644, 7902].forEach((f, i) => this.blip(t + i * 0.07, f, 0.06, 0.14, i * 0.5 - 0.5));
                }
                this.hostCount = payload.distinct_hosts;
            }
            const accepted = payload.images_accepted || 0;
            const rejected = payload.images_rejected || 0;
            if (accepted + rejected > 0) {
                this.gridProbMod = 0.7 + 0.6 * (accepted / (accepted + rejected));
            }
            if (this.enabled && payload.frontier_size != null) {
                const bed = this.profile.bed * this.density * Math.min(2, 0.5 + payload.frontier_size / 20000);
                this.noiseGain.gain.setTargetAtTime(bed, this.ctx.currentTime, 1.0);
            }
        }

        // Hard cut: silence for durMs, then slam back.
        dropout(durMs) {
            if (!this.enabled) return;
            const g = this.master.gain;
            g.cancelScheduledValues(this.ctx.currentTime);
            g.setTargetAtTime(0.0001, this.ctx.currentTime, 0.008);
            setTimeout(() => {
                if (this.ctx) g.setTargetAtTime(this.level, this.ctx.currentTime, 0.015);
            }, durMs);
        }

        // ms until the next 16th-note, for beat-locking visual glitches
        nextGridDelayMs() {
            if (!this.enabled || !this.ctx) return 0;
            return Math.max(0, (this.nextNoteTime - this.ctx.currentTime) * 1000);
        }
    }

    const audio = new AudioEngine();

    // ------------------------------------------------------------------
    // Conductor: scenes, drift, reveals, dropouts, micro-events
    // ------------------------------------------------------------------
    const SCENES = [
        {
            name: 'HALFTONE FIELD',
            audio: { bpm: 126, sweep: 2400, bed: 0.015, gridProb: 0.7, clickProb: 0.2, grainProb: 0.45 },
            hold: [24, 40],
            params: {
                styleA: 2, styleB: 8, styleMix: 0.15, threshold: 0.5, contrast: 1.5, jitter: 0.3,
                dither: 3, blockScale: 14, sliceAmp: 0.1, grid: 0.55,
                moshAmount: 0.15, moshBlock: 0.03, moshDrop: 0.1, moshDecay: 0.97,
                scanline: 0.3, noise: 0.06, strobe: 0, invert: 0,
                fade: 0.55, switchInterval: 0.3, tileFactor: 3, tileFraction: 0.5,
                scrollX: 0.05, scrollY: 0, flashBoost: 0.5, gutter: 0
            }
        },
        {
            name: 'BINARY WALL',
            audio: { bpm: 152, sweep: 5200, bed: 0.02, gridProb: 0.55, clickProb: 0.65, grainProb: 0.3, pitchSet: [2960, 3951, 5274, 7040, 8869] },
            hold: [18, 32],
            params: {
                styleA: 1, styleB: 10, styleMix: 0.35, threshold: 0.48, contrast: 1.8, jitter: 0.55,
                dither: 2, blockScale: 12, sliceAmp: 0.08, grid: 0.35,
                moshAmount: 0.05, moshBlock: 0.02, moshDrop: 0.05, moshDecay: 0.985,
                scanline: 0.25, noise: 0.12, strobe: 0, invert: 0,
                fade: 0.8, switchInterval: 0.12, tileFactor: 4, tileFraction: 0.6,
                scrollX: 0.0, scrollY: 0.03, flashBoost: 0.7, gutter: 0
            }
        },
        {
            name: 'WIREFRAME',
            audio: { bpm: 96, sweep: 900, bed: 0.03, gridProb: 0.4, clickProb: 0.12, grainProb: 0.35, grainPitch: 0.7, pitchSet: [622.3, 932.3, 1244.5, 1864.7] },
            hold: [20, 35],
            params: {
                styleA: 3, styleB: 11, styleMix: 0.3, threshold: 0.5, contrast: 2.2, jitter: 0.2,
                dither: 3, blockScale: 16, sliceAmp: 0.05, grid: 0.7,
                moshAmount: 0.1, moshBlock: 0.05, moshDrop: 0.12, moshDecay: 0.96,
                scanline: 0.5, noise: 0.04, strobe: 0, invert: 0,
                fade: 0.4, switchInterval: 0.45, tileFactor: 2, tileFraction: 0.4,
                scrollX: 0.0, scrollY: -0.02, flashBoost: 0.4, gutter: 0.1
            }
        },
        {
            name: 'MELT',
            audio: { bpm: 74, sweep: 500, bed: 0.045, gridProb: 0.25, clickProb: 0.08, grainProb: 0.8, grainPitch: 0.5, pitchSet: [311.1, 415.3, 622.3, 830.6] },
            hold: [22, 38],
            params: {
                styleA: 0, styleB: 5, styleMix: 0.35, threshold: 0.5, contrast: 1.3, jitter: 0.3,
                dither: 4, blockScale: 18, sliceAmp: 0.35, grid: 0.2,
                moshAmount: 0.6, moshBlock: 0.05, moshDrop: 0.45, moshDecay: 0.93,
                scanline: 0.2, noise: 0.05, strobe: 0, invert: 0,
                fade: 0.25, switchInterval: 0.5, tileFactor: 2, tileFraction: 0.35,
                scrollX: 0.02, scrollY: 0.01, flashBoost: 0.2, gutter: 0
            }
        },
        {
            name: 'READOUT',
            audio: { bpm: 132, sweep: 3600, bed: 0.02, gridProb: 0.95, clickProb: 0.3, grainProb: 0.25 },
            hold: [18, 30],
            params: {
                styleA: 6, styleB: 7, styleMix: 0.4, threshold: 0.4, contrast: 1.6, jitter: 0.25,
                dither: 3, blockScale: 12, sliceAmp: 0.1, grid: 0.8,
                moshAmount: 0.08, moshBlock: 0.025, moshDrop: 0.08, moshDecay: 0.975,
                scanline: 0.45, noise: 0.05, strobe: 0, invert: 0,
                fade: 0.6, switchInterval: 0.4, tileFactor: 2, tileFraction: 0.5,
                scrollX: -0.04, scrollY: 0, flashBoost: 0.5, gutter: 0.15
            }
        },
        {
            name: 'HEX RAIN',
            audio: { bpm: 118, sweep: 1800, bed: 0.025, gridProb: 0.6, clickProb: 0.25, grainProb: 0.5 },
            hold: [20, 34],
            params: {
                styleA: 8, styleB: 8, styleMix: 0.5, threshold: 0.5, contrast: 1.4, jitter: 0.3,
                dither: 3, blockScale: 10, sliceAmp: 0.06, grid: 0.6,
                moshAmount: 0.12, moshBlock: 0.03, moshDrop: 0.15, moshDecay: 0.96,
                scanline: 0.3, noise: 0.08, strobe: 0, invert: 0,
                fade: 0.6, switchInterval: 0.25, tileFactor: 2, tileFraction: 0.45,
                scrollX: 0, scrollY: 0.05, flashBoost: 0.5, gutter: 0
            }
        },
        {
            name: 'AVALANCHE',
            audio: { bpm: 168, sweep: 6400, bed: 0.035, gridProb: 0.8, clickProb: 0.9, grainProb: 0.7, subPattern: true },
            hold: [12, 22],
            params: {
                styleA: 2, styleB: 4, styleMix: 0.5, threshold: 0.5, contrast: 1.7, jitter: 0.5,
                dither: 2, blockScale: 12, sliceAmp: 0.25, grid: 0.4,
                moshAmount: 0.3, moshBlock: 0.02, moshDrop: 0.2, moshDecay: 0.95,
                scanline: 0.35, noise: 0.16, strobe: 0.2, invert: 0,
                fade: 0.95, switchInterval: 0.03, tileFactor: 5, tileFraction: 0.8,
                scrollX: 0.12, scrollY: -0.06, flashBoost: 0.9, gutter: 0
            }
        },
        {
            name: 'STATIC',
            audio: { bpm: 60, sweep: 300, bed: 0.05, gridProb: 0.12, clickProb: 0.05, grainProb: 0.3, grainPitch: 0.4, pitchSet: [4978, 7040, 9956] },
            hold: [14, 24],
            params: {
                styleA: 9, styleB: 1, styleMix: 0.3, threshold: 0.55, contrast: 1.2, jitter: 0.4,
                dither: 6, blockScale: 24, sliceAmp: 0.15, grid: 0.15,
                moshAmount: 0.8, moshBlock: 0.09, moshDrop: 0.7, moshDecay: 0.9,
                scanline: 0.75, noise: 0.1, strobe: 0, invert: 0,
                fade: 0.1, switchInterval: 0.8, tileFactor: 1, tileFraction: 0.3,
                scrollX: 0.005, scrollY: 0, flashBoost: 0.1, gutter: 0
            }
        },
        {
            name: 'ARCHIVE',
            audio: { bpm: 100, sweep: 1500, bed: 0.01, gridProb: 0.3, clickProb: 0.15, grainProb: 0.2 },
            hold: [20, 35],
            params: {
                styleA: 0, styleB: 0, styleMix: 0, threshold: 0.5, contrast: 1.15, jitter: 0.1,
                dither: 3, blockScale: 16, sliceAmp: 0, grid: 0.5,
                moshAmount: 0, moshBlock: 0.02, moshDrop: 0, moshDecay: 0.985,
                scanline: 0.08, noise: 0.02, strobe: 0, invert: 0,
                fade: 0.75, switchInterval: 0.55, tileFactor: 3, tileFraction: 0.35,
                scrollX: 0.01, scrollY: 0, flashBoost: 0.45, gutter: 0.55
            }
        }
    ];

    const conductor = {
        auto: true,
        sceneIndex: 0,
        sceneEndsAt: 0,
        strobeRestore: null,
        reveal: false,
        t: 0,

        cut(index, manual = false) {
            if (debug.active) return;
            this.sceneIndex = ((index % SCENES.length) + SCENES.length) % SCENES.length;
            const scene = SCENES[this.sceneIndex];
            applyParams(scene.params);
            const [lo, hi] = scene.hold;
            this.sceneEndsAt = performance.now() + (lo + Math.random() * (hi - lo)) * 1000;
            engine.pulse(1.0);
            audio.onScene(scene);
            flashScene(scene.name);
            setText('sceneLabel', scene.name);
            const select = document.getElementById('sceneSelect');
            if (select) select.value = String(this.sceneIndex);
            if (manual) console.log(`[conductor] manual cut -> ${scene.name}`);
        },

        next() { this.cut(this.sceneIndex + 1 + Math.floor(Math.random() * (SCENES.length - 1))); },

        // ramp bypass up and back: the raw images surface through the abstraction
        startReveal() {
            if (this.reveal || debug.active) return;
            this.reveal = true;
            const up = 600, hold = 800, down = 1100;
            const t0 = performance.now();
            const timer = setInterval(() => {
                const el = performance.now() - t0;
                let v = 0;
                if (el < up) v = (el / up) * 0.85;
                else if (el < up + hold) v = 0.85;
                else if (el < up + hold + down) v = 0.85 * (1 - (el - up - hold) / down);
                else {
                    clearInterval(timer);
                    this.reveal = false;
                    v = 0;
                }
                if (!debug.active) engine.setBypass(v);
            }, 50);
        },

        // hard cut: audio dropout + visual blackout, then slam back
        dropout() {
            if (debug.active) return;
            const dur = 700 + Math.random() * 800;
            audio.dropout(dur);
            const blackout = document.getElementById('blackout');
            if (blackout) blackout.classList.add('on');
            setTimeout(() => {
                if (blackout) blackout.classList.remove('on');
                engine.pulse(1.0);
                audio.onPulse(1.0);
            }, dur);
        },

        // fire a glitch pulse, locked to the audio grid when sound is on
        firePulse(strength) {
            const delay = audio.nextGridDelayMs();
            setTimeout(() => {
                engine.pulse(strength);
                audio.onPulse(strength);
            }, delay);
        },

        tick(dtMs) {
            if (debug.active) return;
            this.t += dtMs / 1000;

            // slow breathing of threshold/mosh so nothing is ever static
            drift.threshold = 0.06 * Math.sin(this.t * 0.31) + 0.03 * Math.sin(this.t * 1.7);
            drift.moshAmount = 0.05 * Math.sin(this.t * 0.13 + 1.0);
            drift.sliceAmp = 0.04 * Math.sin(this.t * 0.47 + 2.0);
            pushTone(); pushMosh(); pushStructure();

            if (!this.auto) return;

            const p = dtMs / 1000;
            if (Math.random() < p * 0.22) this.firePulse(0.4 + Math.random() * 0.6);
            if (Math.random() < p * 0.03) this.startReveal();
            if (audio.enabled && Math.random() < p * 0.015) this.dropout();

            // rare strobe burst, restored after ~700ms
            if (this.strobeRestore === null && Math.random() < p * 0.02) {
                const prev = state.strobe;
                state.strobe = 0.5 + Math.random() * 0.4;
                pushTemporal();
                this.strobeRestore = setTimeout(() => {
                    state.strobe = prev;
                    pushTemporal();
                    this.strobeRestore = null;
                }, 700);
            }

            if (performance.now() >= this.sceneEndsAt) this.next();
        }
    };

    // ------------------------------------------------------------------
    // Debug / neutral mode: the crawl, unprocessed
    // ------------------------------------------------------------------
    const debug = { active: false, saved: null, wasAuto: true };

    function syncAutoUi() {
        const box = document.getElementById('autoMode');
        if (box) box.checked = conductor.auto;
        setText('autoModeVal', conductor.auto ? 'ON' : 'OFF');
    }

    function setDebugMode(on) {
        if (on === debug.active) return;
        const btn = document.getElementById('debugToggle');
        if (on) {
            debug.saved = { ...state };
            debug.wasAuto = conductor.auto;
            conductor.auto = false;
            syncAutoUi();
            debug.active = true;
            applyParams({
                moshAmount: 0, moshDrop: 0, sliceAmp: 0,
                scanline: 0, noise: 0, strobe: 0, invert: 0,
                grid: 0.35, gutter: 0.6, flashBoost: 0.25, colorBleed: 1,
                fade: 0.9, switchInterval: 0.5, scrollX: 0, scrollY: 0
            });
            engine.setBypass(1.0);
            if (btn) btn.classList.add('active');
            setText('sceneLabel', 'DEBUG');
            flashScene('DEBUG');
        } else {
            debug.active = false;
            if (debug.saved) applyParams(debug.saved);
            engine.setBypass(0.0);
            conductor.auto = debug.wasAuto;
            syncAutoUi();
            if (btn) btn.classList.remove('active');
            setText('sceneLabel', SCENES[conductor.sceneIndex].name);
            if (conductor.auto) conductor.sceneEndsAt = performance.now() + 8000;
        }
    }

    // ------------------------------------------------------------------
    // UI plumbing
    // ------------------------------------------------------------------
    let programmatic = false; // true while conductor/debug writes sliders

    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function fmt(v, digits = 2) {
        return Number(v).toFixed(digits);
    }

    function flashScene(name) {
        const el = document.getElementById('sceneFlash');
        if (!el) return;
        el.textContent = name;
        el.classList.remove('on');
        void el.offsetWidth; // restart animation
        el.classList.add('on');
    }

    const GROUPS = {
        styles: { push: pushStyles, artistic: true },
        tone: { push: pushTone, artistic: true },
        structure: { push: pushStructure, artistic: true },
        mosh: { push: pushMosh, artistic: true },
        temporal: { push: pushTemporal, artistic: true },
        accents: { push: pushAccents, artistic: true },
        flow: { push: pushFlow, artistic: true }
    };

    const CONTROLS = {
        styleMix: { group: 'styles', digits: 2 },
        threshold: { group: 'tone', digits: 2 },
        contrast: { group: 'tone', digits: 2 },
        jitter: { group: 'tone', digits: 2 },
        colorBleed: { group: 'tone', digits: 2 },
        dither: { group: 'structure', digits: 0 },
        blockScale: { group: 'structure', digits: 0 },
        sliceAmp: { group: 'structure', digits: 2 },
        grid: { group: 'structure', digits: 2 },
        flashBoost: { group: 'accents', digits: 2 },
        gutter: { group: 'accents', digits: 2 },
        moshAmount: { group: 'mosh', digits: 2 },
        moshBlock: { group: 'mosh', digits: 3 },
        moshDrop: { group: 'mosh', digits: 2 },
        moshDecay: { group: 'mosh', digits: 3 },
        scanline: { group: 'temporal', digits: 2 },
        noise: { group: 'temporal', digits: 2 },
        strobe: { group: 'temporal', digits: 2 },
        invert: { group: 'temporal', digits: 2 },
        fade: { group: 'flow', digits: 2 },
        switchInterval: { group: 'flow', digits: 2 },
        tileFactor: { group: 'flow', digits: 0 },
        tileFraction: { group: 'flow', digits: 2 },
        scrollX: { group: 'flow', digits: 3 },
        scrollY: { group: 'flow', digits: 3 },
        uploads: { group: 'flow', digits: 0 }
    };

    function disableAuto() {
        if (!conductor.auto) return;
        conductor.auto = false;
        syncAutoUi();
    }

    for (const [id, spec] of Object.entries(CONTROLS)) {
        const slider = document.getElementById(id);
        if (!slider) continue;
        slider.addEventListener('input', () => {
            state[id] = parseFloat(slider.value);
            setText(id + 'Val', fmt(state[id], spec.digits));
            GROUPS[spec.group].push();
            if (!programmatic && GROUPS[spec.group].artistic) disableAuto();
        });
    }

    // style selects
    for (const selectId of ['styleA', 'styleB']) {
        const select = document.getElementById(selectId);
        if (!select) continue;
        STYLE_NAMES.forEach((name, i) => {
            const opt = document.createElement('option');
            opt.value = String(i);
            opt.textContent = `${i} ${name}`;
            select.appendChild(opt);
        });
        select.value = String(state[selectId]);
        select.addEventListener('change', () => {
            state[selectId] = parseInt(select.value, 10);
            pushStyles();
            if (!programmatic) disableAuto();
        });
    }

    // scene select + buttons
    {
        const select = document.getElementById('sceneSelect');
        SCENES.forEach((scene, i) => {
            const opt = document.createElement('option');
            opt.value = String(i);
            opt.textContent = scene.name;
            select.appendChild(opt);
        });
        select.addEventListener('change', () => conductor.cut(parseInt(select.value, 10), true));
        document.getElementById('nextScene').addEventListener('click', () => conductor.next());
        document.getElementById('pulseBtn').addEventListener('click', () => conductor.firePulse(1.0));
        document.getElementById('debugToggle').addEventListener('click', () => setDebugMode(!debug.active));
        const autoBox = document.getElementById('autoMode');
        autoBox.addEventListener('change', () => {
            conductor.auto = autoBox.checked;
            setText('autoModeVal', conductor.auto ? 'ON' : 'OFF');
            if (conductor.auto) {
                if (debug.active) setDebugMode(false);
                conductor.sceneEndsAt = performance.now() + 5000;
            }
        });
    }

    // Reflect a scene's params into state + engine + panel widgets.
    function applyParams(params) {
        programmatic = true;
        for (const [key, value] of Object.entries(params)) {
            state[key] = value;
            const widget = document.getElementById(key);
            if (widget && widget.tagName === 'INPUT') {
                widget.value = value;
                const spec = CONTROLS[key];
                if (spec) setText(key + 'Val', fmt(value, spec.digits));
            } else if (widget && widget.tagName === 'SELECT') {
                widget.value = String(value);
            }
        }
        pushAll();
        programmatic = false;
    }

    // audio controls
    {
        const toggle = document.getElementById('audioToggle');
        const setAudioUi = () => {
            toggle.textContent = audio.enabled ? 'SOUND OFF' : 'SOUND ON';
            toggle.classList.toggle('active', audio.enabled);
            setText('audioLabel', audio.enabled ? 'ON' : 'OFF');
        };
        toggle.addEventListener('click', async () => {
            if (audio.enabled) await audio.stop(); else await audio.start();
            setAudioUi();
        });
        document.getElementById('audioLevel').addEventListener('input', (e) => {
            audio.setLevel(parseFloat(e.target.value));
            setText('audioLevelVal', fmt(e.target.value));
        });
        document.getElementById('audioDensity').addEventListener('input', (e) => {
            audio.setDensity(parseFloat(e.target.value));
            setText('audioDensityVal', fmt(e.target.value));
        });
        window.__toggleAudio = () => toggle.click();
    }

    // keyboard
    document.addEventListener('keydown', (event) => {
        const tag = (event.target.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
        const key = event.key.toLowerCase();
        if (key === 'h') {
            document.getElementById('panel').classList.toggle('open');
        } else if (key === 'a') {
            window.__toggleAudio();
        } else if (key === 'd') {
            setDebugMode(!debug.active);
        } else if (key === ' ') {
            conductor.next();
            event.preventDefault();
        } else if (key === 'p') {
            conductor.firePulse(1.0);
        } else if (key === 'i') {
            const prev = state.invert;
            state.invert = 1;
            pushTemporal();
            setTimeout(() => { state.invert = prev; pushTemporal(); }, 150);
        } else if (key === 'f') {
            if (!document.fullscreenElement) document.documentElement.requestFullscreen();
            else document.exitFullscreen();
        } else if (key >= '0' && key <= '9') {
            const idx = parseInt(key, 10);
            if (idx < SCENES.length) conductor.cut(idx, true);
        }
    });

    // ------------------------------------------------------------------
    // Artifact stream (wire protocol unchanged)
    // ------------------------------------------------------------------
    let ws = null;
    let reconnectAttempt = 0;
    let rendererError = false;
    const deliveryCounters = { received: 0, decoded: 0, gpu_uploaded: 0, presented: 0, rejected: 0, skipped: 0 };

    function updateDeliveryCounter(stage) {
        const id = {
            received: 'receivedCounter',
            decoded: 'decodedCounter',
            gpu_uploaded: 'uploadedCounter',
            presented: 'presentedCounter'
        }[stage];
        if (id) setText(id, deliveryCounters[stage]);
    }

    function acknowledge(stage, sequence) {
        deliveryCounters[stage]++;
        updateDeliveryCounter(stage);
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ack', stage, sequence, at_ms: Date.now() }));
        }
    }

    Module['onRendererEvent'] = (stageCode, sequence) => {
        const stage = { 1: 'decoded', 2: 'gpu_uploaded', 3: 'presented', 4: 'rejected' }[stageCode];
        if (stage) acknowledge(stage, sequence >>> 0);
    };

    Module['onWebGPUError'] = (type, message) => {
        rendererError = true;
        console.error(`WebGPU error ${type}: ${message}`);
        updateStatusBar('GPU ERROR', '#f44');
    };

    function updateStatusBar(status, color) {
        const el = document.getElementById('connectionStatus');
        if (rendererError && status !== 'GPU ERROR') return;
        el.textContent = status;
        el.style.color = color || '#fff';
    }

    function handleArtifactFrame(data) {
        const bytes = new Uint8Array(data);
        if (bytes.length < 4) throw new Error('Artifact frame is too short');
        const headerLength = new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
        if (headerLength <= 0 || headerLength > 65536 || 4 + headerLength > bytes.length) {
            throw new Error('Invalid artifact header length');
        }
        const header = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + headerLength)));
        if (header.protocol !== 1) throw new Error(`Unsupported artifact protocol ${header.protocol}`);
        const payload = bytes.subarray(4 + headerLength);
        if (header.byte_size !== payload.length) throw new Error('Artifact payload length mismatch');
        acknowledge('received', header.sequence);
        if (header.kind !== 'image') {
            acknowledge('skipped', header.sequence);
            return;
        }

        const provenance = document.getElementById('artifactProvenance');
        if (provenance) {
            const rights = header.rights || {};
            provenance.textContent = `#${header.sequence} ${header.producer} | ${header.source_url || 'generated'} | ${rights.license || rights.status || 'unknown rights'}`;
        }

        // each arriving artifact injects event energy, sound, and its sequence
        engine.setSequence(header.sequence >>> 0);
        engine.pulse(0.12);
        audio.onArtifact(header, payload);

        const ptr = Module._malloc(payload.length);
        Module.HEAPU8.set(payload, ptr);
        Module.ccall(
            'onArtifactReceived',
            null,
            ['number', 'number', 'number'],
            [ptr, payload.length, header.sequence >>> 0]
        );
        Module._free(ptr);
    }

    async function connectImageStream() {
        updateStatusBar('CONNECTING', '#ff0');
        ws = new WebSocket(await getImageWebSocketUrl());
        ws.binaryType = 'arraybuffer';
        ws.onopen = () => {
            reconnectAttempt = 0;
            updateStatusBar('LIVE', '#fff');
        };
        ws.onerror = () => updateStatusBar('WS ERROR', '#f44');
        ws.onmessage = (event) => {
            try {
                handleArtifactFrame(event.data);
            } catch (error) {
                console.error('Rejected artifact frame:', error);
                updateStatusBar('PROTOCOL ERROR', '#f44');
            }
        };
        ws.onclose = () => {
            updateStatusBar('RECONNECTING', '#ff0');
            const delay = Math.min(30000, 500 * (2 ** reconnectAttempt));
            reconnectAttempt++;
            setTimeout(() => void connectImageStream(), delay);
        };
    }

    // ------------------------------------------------------------------
    // Crawler steering (endpoints unchanged)
    // ------------------------------------------------------------------
    function parseListInput(value) {
        return value.split(',').map((item) => item.trim()).filter(Boolean);
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
            return `<div><b>${time} ${type}</b>: ${message}</div>`;
        });
        if (errors.length > 0 && recentEvents.length === 0) {
            rows.push(...errors.slice(-5).reverse().map((error) => `<div><b>error</b>: ${escapeHtml(error)}</div>`));
        }
        log.innerHTML = rows.join('');
    }

    function renderCrawlerState(stateResponse) {
        const payload = stateResponse.state || stateResponse;
        const keywords = (payload.keywords || []).join(', ') || 'none';
        const broker = payload.broker || {};
        const status = `keywords: ${keywords} | explore: ${(payload.exploration ?? 0).toFixed(2)}${payload.autopilot ? ' auto' : ''} | domains: ${payload.distinct_hosts ?? '--'} | frontier: ${payload.frontier_size ?? '--'} | media: ${payload.media_queue_size ?? '--'} | broker: ${broker.broker_resident ?? payload.queue_size ?? '--'}/${broker.broker_capacity ?? '--'} | pages: ${payload.pages_visited ?? '--'} | accepted: ${payload.images_accepted ?? '--'} | rejected: ${payload.images_rejected ?? '--'} | duplicate: ${payload.images_duplicate ?? '--'}`;
        setText('crawlerStatus', status);
        setText('crawlerStatusLabel', `${payload.queue_size ?? 0}/${payload.images_accepted ?? 0}`);
        const exploration = document.getElementById('crawlerExploration');
        if (exploration && document.activeElement !== exploration) exploration.value = payload.exploration ?? 0.55;
        setText('crawlerExplorationValue', Number(payload.exploration ?? 0.55).toFixed(2));
        const autopilot = document.getElementById('crawlerAutopilot');
        if (autopilot) autopilot.checked = Boolean(payload.autopilot);
        const contentPolicy = document.getElementById('crawlerContentPolicy');
        if (contentPolicy) contentPolicy.value = payload.content_policy ?? broker.content_policy ?? 'broad';
        renderCrawlerLog(payload.recent_events || [], payload.recent_errors || []);
        audio.onTelemetry(payload);
    }

    async function refreshCrawlerState() {
        try {
            renderCrawlerState(await crawlerRequest('/api/crawler/state'));
        } catch (error) {
            setText('crawlerStatus', `Crawler: ${error.message}`);
            setText('crawlerStatusLabel', 'offline');
            setText('crawlerLog', `Crawler log: ${error.message}`);
        }
    }

    async function setupCrawlerControls() {
        const keywordInput = document.getElementById('crawlerKeywords');
        const seedInput = document.getElementById('crawlerSeed');
        const keywordButton = document.getElementById('applyCrawlerKeywords');
        const seedButton = document.getElementById('addCrawlerSeed');
        const exploration = document.getElementById('crawlerExploration');
        const autopilot = document.getElementById('crawlerAutopilot');
        const contentPolicy = document.getElementById('crawlerContentPolicy');
        const runtimeConfig = await getRuntimeConfig();

        if (!runtimeConfig.crawler_enabled) {
            [keywordInput, seedInput, keywordButton, seedButton, exploration, autopilot, contentPolicy]
                .filter(Boolean)
                .forEach((control) => { control.disabled = true; });
            setText('crawlerStatus', `Crawler controls inactive in ${runtimeConfig.mode || 'this'} mode`);
            setText('crawlerStatusLabel', 'inactive');
            setText('crawlerLog', 'Start with --web-crawler to enable the autonomous journey.');
            return;
        }

        keywordButton.addEventListener('click', async () => {
            try {
                const keywords = parseListInput(keywordInput.value);
                renderCrawlerState(await crawlerRequest('/api/crawler/keywords', {
                    method: 'POST',
                    body: JSON.stringify({ keywords })
                }));
            } catch (error) {
                setText('crawlerStatus', `Crawler: ${error.message}`);
            }
        });

        seedButton.addEventListener('click', async () => {
            try {
                const seed = seedInput.value.trim();
                renderCrawlerState(await crawlerRequest('/api/crawler/seeds', {
                    method: 'POST',
                    body: JSON.stringify({ seeds: seed ? [seed] : [] })
                }));
            } catch (error) {
                setText('crawlerStatus', `Crawler: ${error.message}`);
            }
        });

        exploration.addEventListener('input', () => {
            setText('crawlerExplorationValue', Number(exploration.value).toFixed(2));
        });
        exploration.addEventListener('change', async () => {
            try {
                renderCrawlerState(await crawlerRequest('/api/crawler/exploration', {
                    method: 'POST',
                    body: JSON.stringify({ exploration: Number(exploration.value) })
                }));
            } catch (error) {
                setText('crawlerStatus', `Crawler: ${error.message}`);
            }
        });

        autopilot.addEventListener('change', async () => {
            try {
                renderCrawlerState(await crawlerRequest('/api/crawler/autopilot', {
                    method: 'POST',
                    body: JSON.stringify({ enabled: autopilot.checked })
                }));
            } catch (error) {
                setText('crawlerStatus', `Crawler: ${error.message}`);
            }
        });

        contentPolicy.addEventListener('change', async () => {
            try {
                renderCrawlerState(await crawlerRequest('/api/crawler/content-policy', {
                    method: 'POST',
                    body: JSON.stringify({ policy: contentPolicy.value })
                }));
            } catch (error) {
                setText('crawlerStatus', `Crawler: ${error.message}`);
            }
        });

        refreshCrawlerState();
        setInterval(refreshCrawlerState, 3000);
    }

    // ------------------------------------------------------------------
    // Boot
    // ------------------------------------------------------------------
    void connectImageStream();
    void setupCrawlerControls();

    conductor.cut(0);
    let lastTick = performance.now();
    setInterval(() => {
        const now = performance.now();
        conductor.tick(now - lastTick);
        lastTick = now;
    }, 250);

    setInterval(() => {
        setText('bufferUsageLabel', `${engine.getBufferUsage()}/${engine.getRingBufferSize()}`);
    }, 2000);

    console.log('DATAVALANCHE: H panel, A sound, D debug, SPACE cut, P pulse, I invert, F fullscreen, 0-8 scenes');
};

Module['onAbort'] = (what) => {
    Module.runtimeAbortReason = String(what);
    console.error('WebAssembly module aborted:', what);
    const el = document.getElementById('connectionStatus');
    if (el) el.textContent = 'WASM ERROR';
};
