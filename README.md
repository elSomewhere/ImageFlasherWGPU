# ImageFlasherWGPU

An interactive digital art piece that viscerally demonstrates **information overload** in our hyperconnected world. Using WebGPU for high-performance rendering, it creates an overwhelming stream of images that flash by faster than human comprehension—mirroring how we consume digital content in the internet age.

The project now includes an autonomous, compliance-first web journey: starting from
your seed URLs, it endlessly follows public links, collects and normalizes media into
a bounded rolling buffer, and replays the evolving digital world through a
GPU-resident image wall. Keywords, seed URLs, exploration, autopilot, and content
policy remain steerable at runtime; autonomous Wikimedia seeders are optional
plugins.

> 📖 **[Read the full conceptual framework](CONCEPT.md)** to understand the artistic vision and cultural commentary behind this project.
>
> 🧭 **[Read the crawler architecture](CRAWLER_ARCHITECTURE.md)** for the traversal,
> compliance, backpressure, protocol, renderer, and extension design.

![Demo](https://img.shields.io/badge/WebGPU-Powered-brightgreen) ![Platform](https://img.shields.io/badge/Platform-Web-blue) ![Language](https://img.shields.io/badge/Language-C%2B%2B%2FPython%2FNode.js-orange) ![Art](https://img.shields.io/badge/Purpose-Digital_Art-purple)

## ✨ DATAVALANCHE Presentation Layer

The presentation layer renders the crawl as a monochrome data avalanche in the register of Ryoji Ikeda / raster-noton:

- 🎨 **10 Tile Materials**: every tile re-materializes its image as halftone, wireframe, pixel-sort smear, waveform readout, barcode, hex-glyph rain, mosaic, or hard 1-bit data
- 🧨 **Datamosh Feedback**: block-displacement + P-frame-drop feedback loop; arriving artifacts inject glitch energy
- 🔊 **Data Sonification**: clicks per artifact, raw image bytes as PCM, analysis-pitched sine grid, sub pulses — the crawl made audible
- 🎼 **Autonomous Conductor**: scenes cut and drift on their own; the piece runs config-free but stays fully steerable
- 🖤 **Strict Monochrome**: hairline grids, scanlines, binary timecode, strobe — the interface disappears behind the work

## 🚀 Quick Start

### Prerequisites

1. **Node.js 18+** and **Python 3.11+**
2. **A current Emscripten SDK** (the build downloads its official Dawn WebGPU port)
3. **Python packages** (installed into a project-local virtual environment):
   ```bash
   make setup-python
   ```

### Installation & Running

```bash
# Clone and setup
git clone https://github.com/elSomewhere/ImageFlasherWGPU.git
cd ImageFlasherWGPU
npm install
make setup-python

# Build (if needed) and start
npm run build  # Only needed if WASM files are missing
npm start      # Start the Ikeda interface (default)

# Alternative modes
npm run start:reddit     # Reddit image scraping
npm run start:generated  # Generated VHS-style images
npm run start:web-crawler # Autonomous broad-web journey
```

### Direct Node.js Usage

```bash
# Default: Ikeda data visualization mode
node server.js

# Reddit images from specific subreddit
node server.js --reddit --subreddit cats
node server.js --reddit --subreddit cyberpunk

# Generated images mode
node server.js --generated

# Link-only endless walk from your seed URLs
node server.js --web-crawler --seeds https://blick.ch/
node server.js --web-crawler --seeds https://blick.ch/ --keywords brutalism,astronomy

# Optional: autonomous Wikimedia seeders and the Commons media lane
node server.js --web-crawler --seed-plugins wikipedia_random,wikidata_official_sites --enable-commons
```

## 🎮 Interface Usage

### Web Interface
Open `http://localhost:8000` for the **DATAVALANCHE** presentation. The canvas is the interface; controls stay hidden until requested:

- **Conductor**: autonomous scene evolution (default ON) — scenes cut every 12–45 s, parameters breathe continuously, glitch pulses fire stochastically. Manual edits to any material/tone slider take over and disable auto.
- **Panel** (`H`): scene select, tile materials, tone, structure, datamosh, temporal, flow, audio, and crawler steering.
- **Sonification** (`A` or SOUND button): Ikeda-style clicks per crawled artifact, raw image bytes played as PCM bursts, a sine grid pitched by per-image analysis (luminance→pitch, entropy→duration), sub pulses on scene cuts — all through a limiter.
- **Crawler Steering**: exploration/autopilot, keywords, seeds, content policy, live queue metrics.
- **Delivery Telemetry**: received, decoded, GPU-uploaded, and presented artifact counts in the status strip.

### Keyboard Shortcuts (Exhibition Ready)

| Key | Action |
|-----|--------|
| `H` | Toggle control panel |
| `A` | Toggle sonification |
| `D` | Debug / neutral mode: images exactly as collected |
| `SPACE` | Cut to a new scene |
| `P` | Fire a glitch pulse |
| `I` | Momentary invert flash |
| `F` | Toggle fullscreen |
| `0-8` | Jump to scene by index |

## 🎨 Presentation Layer

### Tile materials
Each tile renders its ring-buffer image through one of 10 monochrome materials; every tile picks between the two active styles by stable hash, so the wall is heterogeneous but coherent:

| # | Material | Treatment |
|---|----------|-----------|
| 0 | RAW | grayscale luma |
| 1 | THRESH | 1-bit threshold, per-tile jitter |
| 2 | BAYER | ordered-dither halftone |
| 3 | EDGE | Sobel wireframe on black |
| 4 | SORT | luma-keyed pixel-sort smear |
| 5 | SLICE | displaced bands + posterize |
| 6 | WAVE | image rows redrawn as waveform bars |
| 7 | BARCODE | columns collapsed to stripes |
| 8 | HEX | image blocks printed as hex glyphs |
| 9 | BLOCKS | hard mosaic with dropout |
| 10 | BITPLANE | single extracted bit-plane of luma |
| 11 | CONTOUR | quantized luma iso-lines |

### Datamosh feedback
The frame-blend pass is a mosh engine: the previous frame is re-sampled through block displacement ("broken motion vectors", re-rolled 7×/s) and per-block **P-frame drops** that hold stale image data with luminance decay. Arriving artifacts and scene cuts inject event energy that spikes the mosh and shears the frame.

### Global composition
Scanlines + rolling sync bar, hairline grid, sparse bit-flip noise, a binary strip counting the clock (left) and the crawl sequence (right), strobe/invert, hard mono enforcement (optional color bleed). Fresh tile switches flash white (age-driven), tiles can carry contact-sheet gutters and ring-slot stamps. Scenes: HALFTONE FIELD, BINARY WALL, WIREFRAME, MELT, READOUT, HEX RAIN, AVALANCHE, STATIC, ARCHIVE. The conductor also fires REVEAL moments (the raw images momentarily surface through the abstraction) and hard blackout/dropout cuts.

### Debug / neutral mode
`D` (or the DEBUG button) bypasses the whole treatment chain: original color images exactly as collected, tiled as a contact sheet with gutters and slot stamps, no mosh, no overlays — a live view for verifying what the crawler brings home. Toggling back restores the previous scene and auto-evolution.

### Sonification
All layers run behind a limiter and follow per-scene sound profiles: clicks per crawled artifact (stereo-placed by sequence), a **granular texture cut live from the raw bytes of recent artifacts**, an analysis-pitched sine grid (luminance→pitch, entropy→duration), sub pulses and a pitch-dropping kick on scene cuts, plus crawl-telemetry sonics — new-domain three-tone pings, frontier size driving the noise bed, and the accept/reject ratio thinning the grid. Visual glitches quantize to the audio's 16th-note grid when sound is on.

### Render parameter API (WASM exports)
`setStyles(a,b,prob)` · `setTone(threshold,contrast,colorBleed,jitter)` · `setStructure(dither,block,sliceAmp,grid)` · `setMosh(amount,block,drop,decay)` · `setTemporal(scanline,noise,strobe,invert)` · `setAccents(flash,gutter)` · `setBypass(mix)` · `setSequence(seq)` · `pulse(strength)` plus the flow controls (`setFadeFactor`, `setImageSwitchInterval`, `setTileFactor`, `setRandomTileFraction`, `setScrollingSpeed`, `setMaxUploadsPerFrame`). All share one 112-byte `RenderParams` uniform bound in every pass.

## 🏗️ Development

### Project Structure

```
├── public/               # Web interface (served by Node.js)
│   ├── index.html       # Enhanced Ikeda interface
│   ├── app.js          # WebGPU client code
│   ├── index.js        # Generated WASM module
│   └── index.wasm      # WebAssembly binary
├── src/
│   ├── cpp/            # C++ WebGPU source code
│   │   ├── main.cpp    # Main application
│   │   └── *.h         # STB image libraries
│   └── python/         # Python image servers
│       ├── ImageCreator_Ikeda.py  # Ikeda data server
│       ├── scraper_3.py           # Reddit scraper
│       ├── web_crawler_server.py  # Autonomous crawler entry point
│       └── crawler/               # Ports/adapters/core/runtime architecture

├── server.js           # Node.js server (main entry point)
├── package.json        # Node.js dependencies & scripts
└── CMakeLists.txt      # Emscripten build configuration
```

### Building from Source

```bash
# Build WebAssembly modules
npm run build:wasm

# Development cycle
npm run dev    # Build and start

# Clean rebuild
npm run rebuild
```

### Available NPM Scripts

| Script | Description |
|--------|-------------|
| `npm start` | Start Ikeda interface |
| `npm run start:reddit` | Start with Reddit scraper |
| `npm run start:generated` | Start with generated images |
| `npm run start:web-crawler` | Start the autonomous crawler |
| `npm run build` | Build WASM modules |
| `npm run dev` | Development build & start |
| `npm run clean` | Clean build artifacts |
| `npm run rebuild` | Full rebuild cycle |
| `npm test` | Run the crawler and protocol regression suite |

## 🔧 Technical Architecture

- **Frontend**: WebGPU shaders (WGSL) + Enhanced Ikeda interface
- **Backend**: Node.js + Express server
- **Image Processing**: Python WebSocket servers with real-time analysis
- **Graphics**: C++ compiled to WebAssembly via Emscripten
- **Data Pipeline**: Live statistical analysis with metadata streaming
- **Crawler**: Async page/media workers, host-stratified frontier, novelty autopilot
- **Compliance**: robots.txt, Crawl-delay, SSRF/redirect checks, bounded fetches, backoff
- **Artifact Bus**: Versioned provenance frames, rolling broker, per-viewer GPU credits
- **Renderer**: 256-layer texture-array ring and a single instanced tile draw

## 📊 Data Analysis Features

The enhanced interface provides real-time analysis:

- **Luminance Statistics**: Mean brightness and distribution
- **Information Entropy**: Shannon entropy and complexity metrics
- **Spatial Analysis**: Edge density and texture uniformity  
- **Frequency Analysis**: FFT energy and spectral ratios
- **Compression Estimation**: Data complexity approximation

## 🌐 Browser Compatibility

Requires a modern browser with WebGPU support:
- Chrome 113+
- Firefox 113+
- Safari 16.4+ (macOS 13+)

## 📝 License

This project combines multiple technologies and is intended for educational and artistic purposes.

---

**Experience the intersection of data, art, and technology. 🌆✨**
