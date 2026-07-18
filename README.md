# ImageFlasherWGPU

An interactive digital art piece that viscerally demonstrates **information overload** in our hyperconnected world. Using WebGPU for high-performance rendering, it creates an overwhelming stream of images that flash by faster than human comprehension—mirroring how we consume digital content in the internet age.

The project now includes an autonomous, compliance-first web journey: it continuously
traverses public links, collects and normalizes media into a bounded rolling buffer,
and replays the evolving digital world through a GPU-resident image wall. It runs
without configuration, while keywords, seed URLs, exploration, autopilot, and content
policy remain steerable at runtime.

> 📖 **[Read the full conceptual framework](CONCEPT.md)** to understand the artistic vision and cultural commentary behind this project.
>
> 🧭 **[Read the crawler architecture](CRAWLER_ARCHITECTURE.md)** for the traversal,
> compliance, backpressure, protocol, renderer, and extension design.

![Demo](https://img.shields.io/badge/WebGPU-Powered-brightgreen) ![Platform](https://img.shields.io/badge/Platform-Web-blue) ![Language](https://img.shields.io/badge/Language-C%2B%2B%2FPython%2FNode.js-orange) ![Art](https://img.shields.io/badge/Purpose-Digital_Art-purple)

## ✨ Enhanced Ikeda Interface

This project now features an **enhanced Ryoji Ikeda-inspired interface** with:

- 🎨 **13 Visual Modes**: BLACK/WHITE, GRID, DATA, BINARY, FREQUENCY, SCAN, MATRIX, PULSE, NOISE, STRIP, PHASE, QUANTUM
- 📊 **Real-time Data Analysis**: Live image statistics, entropy, variance, edge density
- ⌨️ **Exhibition-Ready Controls**: Instant keyboard shortcuts for seamless performance
- 🖤 **Minimalist Aesthetic**: Pure black/white interface matching Ikeda's design philosophy
- 📡 **Live Data Visualization**: Transform images into mathematical representations

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

# Autonomous crawler; seeds/keywords are optional
node server.js --web-crawler
node server.js --web-crawler --keywords brutalism,astronomy
```

## 🎮 Interface Usage

### Web Interface
Open `http://localhost:8000` to access the **Data.Matrix** interface featuring:

- **Visual Mode Selector**: Switch between 13 distinct processing modes
- **Core Processing**: Threshold, grid size, data intensity controls
- **Mode Parameters**: Frequency, scan speed, matrix scale, pulse rate, etc.
- **Live Analysis**: Real-time luminance, entropy, variance, edge density
- **Motion Controls**: Scrolling speed and offset adjustments
- **Crawler Steering**: Exploration/autopilot, keywords, seeds, content policy, live queue metrics
- **Delivery Telemetry**: Received, decoded, GPU-uploaded, and presented artifact counts

### Keyboard Shortcuts (Exhibition Ready)

| Key | Action |
|-----|--------|
| `1-9, 0, -, =` | Switch to modes 1-13 |
| `B` | Black/White mode |
| `G` | Grid mode |
| `D` | Data mode |
| `F` | Frequency mode |
| `S` | Scan mode |
| `M` | Matrix mode |
| `P` | Pulse mode |
| `N` | Noise mode |
| `T` | Cycle threshold values |
| `R` | Reset to defaults |
| `ESC` | Toggle fullscreen |

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

## 🎨 Visual Modes

### Core Modes
- **BLACK/WHITE** (1): Pure binary representation
- **GRID** (2): Geometric decomposition  
- **DATA** (3): Statistical visualization
- **BINARY** (4): Digital encoding display

### Advanced Processing
- **FREQUENCY** (5): Spectral analysis visualization
- **SCAN** (6): Progressive image scanning
- **MATRIX** (7): Mathematical transformation
- **PULSE** (8): Rhythmic intensity modulation
- **NOISE** (9): Entropy-based pattern generation
- **STRIP** (10): Linear decomposition
- **PHASE** (11): Phase shift visualization
- **QUANTUM** (12): Discrete state representation

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
