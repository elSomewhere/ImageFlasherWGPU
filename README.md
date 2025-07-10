# ImageFlasherWGPU

An interactive digital art piece that viscerally demonstrates **information overload** in our hyperconnected world. Using WebGPU for high-performance rendering, it creates an overwhelming stream of images that flash by faster than human comprehension—mirroring how we consume digital content in the internet age.

> 📖 **[Read the full conceptual framework](CONCEPT.md)** to understand the artistic vision and cultural commentary behind this project.

![Demo](https://img.shields.io/badge/WebGPU-Powered-brightgreen) ![Platform](https://img.shields.io/badge/Platform-Web-blue) ![Language](https://img.shields.io/badge/Language-C%2B%2B%2FPython-orange) ![Art](https://img.shields.io/badge/Purpose-Digital_Art-purple)

## Features

- 🎮 **WebGPU Rendering**: High-performance GPU-accelerated graphics
- 🎨 **VHS Aesthetic**: Retro glitch effects and scan lines
- 🔄 **Real-time Effects**: Adjustable fading, tiling, and scrolling
- 🌐 **Web Interface**: Cyberpunk-themed control panel
- 📱 **Multiple Image Sources**: Generated art or Reddit scraped images
- ⚡ **Live Controls**: Adjust all effects in real-time

## Quick Start

### Prerequisites

1. **Python 3.7+** with required packages:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Pre-built WebAssembly files** (already included in `cmake-build-emscripten/`)

### Running the Application

The easiest way to run the application is with the launcher script:

```bash
# Default mode - Generated VHS-style images
python3 launcher.py

# Generated images (explicit)
python3 launcher.py --generated

# Reddit scraped images (worldnews subreddit)
python3 launcher.py --reddit

# Custom subreddit
python3 launcher.py --reddit --subreddit cats
python3 launcher.py --reddit --subreddit art
python3 launcher.py --reddit --subreddit cyberpunk
```

### What happens when you run it:

1. **Web Server**: Starts on `http://localhost:8000`
2. **Image Stream**: WebSocket server on port `5010`
3. **Browser**: Open `http://localhost:8000` to see the application

## Usage

### Control Panel

The web interface includes a control panel with the following adjustable parameters:

- **Fade Factor** (0-1): Controls image transition smoothness
- **Switch Interval** (0.1-5s): How fast images change
- **Tile Factor** (0-4): Number of image tiles displayed
- **Scroll Speed X/Y** (-1 to 1): Image scrolling velocity  
- **Scroll Offset X/Y** (0-1): Starting scroll position
- **Max Uploads/frame** (0-10): Performance throttling (0 = unlimited)

### Image Sources

#### Generated Images (`--generated`)
- Creates random geometric art with VHS effects
- Colorful gradients and shapes
- Fast generation, no network required
- Perfect for testing and demos

#### Reddit Images (`--reddit`)
- Scrapes real images from specified subreddit
- Applies VHS distortion effects
- Resizes to 512x512 automatically
- Popular subreddits to try: `art`, `pics`, `cyberpunk`, `cats`, `nature`

## Advanced Usage

### Manual Setup (if you prefer)

1. **Start Web Server**:
   ```bash
   cd cmake-build-emscripten
   python3 serve.py
   ```

2. **Start Image Server** (choose one):
   ```bash
   # Generated images
   python3 ImageCreator.py
   
   # Reddit scraper  
   python3 scraper_3.py
   ```

3. **Open Browser**: Navigate to `http://localhost:8000`

### Building from Source

If you need to rebuild the WebAssembly:

```bash
# Install Emscripten SDK first
mkdir build && cd build
emcmake cmake ..
emmake make
```

## Technical Details

### Architecture
- **Frontend**: HTML5 Canvas + WebGPU shaders (WGSL)
- **Backend**: C++ compiled to WebAssembly via Emscripten  
- **Image Pipeline**: Python WebSocket servers
- **Graphics**: Real-time GPU compute shaders for effects

### Dependencies
- **C++**: WebGPU, STB Image libraries
- **Python**: PIL, OpenCV, Beautiful Soup, WebSockets, Requests
- **Web**: Modern browser with WebGPU support

### File Structure
```
├── launcher.py           # 🚀 Main launcher script
├── main.cpp             # C++ WebGPU application
├── ImageCreator.py      # Generated images server
├── scraper_3.py         # Reddit scraper server  
├── index.html           # Web interface
├── app.js               # JavaScript WebSocket client
├── requirements.txt     # Python dependencies
└── cmake-build-emscripten/  # Pre-built WebAssembly
```

## Troubleshooting

### Common Issues

**"WebSocket connection failed"**
- Make sure the image server is running on port 5010
- Check that no firewall is blocking the connection

**"cmake-build-emscripten directory not found"**
- The project comes pre-built, but if missing, rebuild with Emscripten

**"Missing required dependency"**  
- Run: `pip3 install -r requirements.txt`

**Browser compatibility**
- Requires a modern browser with WebGPU support (Chrome 113+, Firefox 113+)

### Performance Tips

- Use `--generated` mode for best performance
- Adjust "Max Uploads/frame" if experiencing lag
- Lower the "Switch Interval" for smoother transitions
- Try different tile factors based on your GPU

## Contributing

Feel free to experiment with:
- New shader effects in `main.cpp`
- Additional image sources in Python
- UI improvements in `index.html`
- New subreddit scrapers

## License

This project combines multiple technologies and is intended for educational and artistic purposes.

---

**Enjoy the cyberpunk visual experience! 🌆✨** 