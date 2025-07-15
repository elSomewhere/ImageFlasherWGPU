# ImageFlasherWGPU Makefile
# Comprehensive build and server management

# Configuration
BUILD_DIR = cmake-build-emscripten
WEB_PORT = 8000
WEBSOCKET_PORT = 5010

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m # No Color

.PHONY: help stop build start rebuild clean status ikeda start-ikeda rebuild-ikeda reddit start-reddit rebuild-reddit

# Default target
help:
	@echo "$(GREEN)ImageFlasherWGPU Makefile$(NC)"
	@echo "================================"
	@echo "Available targets:"
	@echo "  $(YELLOW)rebuild$(NC)        - Stop servers, rebuild project, and restart (recommended)"
	@echo "  $(YELLOW)rebuild-ikeda$(NC)   - Rebuild and start Ikeda version"
	@echo "  $(YELLOW)rebuild-reddit$(NC)  - Rebuild and start Reddit crawler version"
	@echo "  $(YELLOW)stop$(NC)           - Stop running servers"
	@echo "  $(YELLOW)build$(NC)          - Rebuild the Emscripten WebAssembly project"
	@echo "  $(YELLOW)start$(NC)          - Start regular application servers"
	@echo "  $(YELLOW)start-ikeda$(NC)    - Start Ikeda version application"
	@echo "  $(YELLOW)start-reddit$(NC)   - Start Reddit crawler application"
	@echo "  $(YELLOW)clean$(NC)          - Clean build artifacts"
	@echo "  $(YELLOW)status$(NC)         - Check server status"
	@echo "  $(YELLOW)help$(NC)           - Show this help message"
	@echo ""
	@echo "$(GREEN)Versions:$(NC)"
	@echo "  $(YELLOW)Regular$(NC):     Visual effects and generated image streaming"
	@echo "  $(YELLOW)Ikeda$(NC):       Minimalist data aesthetics (Ryoji Ikeda inspired)"
	@echo "  $(YELLOW)Reddit$(NC):      Live Reddit image streaming with visual effects"

# Full rebuild cycle (recommended target)
rebuild: stop build start
	@echo "$(GREEN)✅ Full rebuild completed successfully!$(NC)"

# Full rebuild cycle for Ikeda version
rebuild-ikeda: stop build start-ikeda
	@echo "$(GREEN)✅ Ikeda rebuild completed successfully!$(NC)"

# Full rebuild cycle for Reddit version
rebuild-reddit: stop build start-reddit
	@echo "$(GREEN)✅ Reddit rebuild completed successfully!$(NC)"

# Stop running servers
stop:
	@echo "$(YELLOW)🛑 Stopping servers...$(NC)"
	@-pkill -f "node server.js" 2>/dev/null || true
	@-pkill -f "scraper" 2>/dev/null || true
	@-lsof -ti:$(WEB_PORT) | xargs kill 2>/dev/null || true
	@-lsof -ti:$(WEBSOCKET_PORT) | xargs kill 2>/dev/null || true
	@sleep 2
	@echo "$(GREEN)✅ Servers stopped$(NC)"

# Build the Emscripten project
build:
	@echo "$(YELLOW)🔨 Building Emscripten project...$(NC)"
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "$(YELLOW)📁 Creating build directory...$(NC)"; \
		mkdir -p $(BUILD_DIR); \
	fi
	@echo "$(YELLOW)⚙️  Configuring CMake...$(NC)"
	@emcmake cmake -B $(BUILD_DIR) -S .
	@echo "$(YELLOW)🏗️  Compiling WebAssembly...$(NC)"
	@emmake make -C $(BUILD_DIR)
	@echo "$(GREEN)✅ Build completed successfully!$(NC)"

# Start the regular application
start:
	@echo "$(YELLOW)🚀 Starting ImageFlasherWGPU (Regular Version)...$(NC)"
	@echo "$(YELLOW)🌐 Ikeda interface: http://localhost:$(WEB_PORT)$(NC)"
	@echo "$(YELLOW)📡 WebSocket server will run on port: $(WEBSOCKET_PORT)$(NC)"
	@echo "$(YELLOW)⏹️  Press Ctrl+C to stop when ready$(NC)"
	@echo ""
	@node server.js --generated

# Start the Ikeda version (enhanced data aesthetics)
start-ikeda:
	@echo "$(YELLOW)🚀 Starting ImageFlasherWGPU (Ikeda Version)...$(NC)"
	@echo "$(YELLOW)🎨 Ikeda interface: http://localhost:$(WEB_PORT)$(NC)"
	@echo "$(YELLOW)📡 WebSocket server will run on port: $(WEBSOCKET_PORT)$(NC)"
	@echo "$(YELLOW)✨ Features: Data matrix visualization, B/W modes, grid analysis$(NC)"
	@echo "$(YELLOW)⏹️  Press Ctrl+C to stop when ready$(NC)"
	@echo ""
	@node server.js --ikeda

# Start the Reddit crawler version
start-reddit:
	@echo "$(YELLOW)🚀 Starting ImageFlasherWGPU (Reddit Crawler Version)...$(NC)"
	@echo "$(YELLOW)🌐 Ikeda interface: http://localhost:$(WEB_PORT)$(NC)"
	@echo "$(YELLOW)📡 WebSocket server will run on port: $(WEBSOCKET_PORT)$(NC)"
	@echo "$(YELLOW)🔍 Source: Live Reddit image scraping$(NC)"
	@echo "$(YELLOW)⏹️  Press Ctrl+C to stop when ready$(NC)"
	@echo ""
	@node server.js --reddit

# Start Reddit crawler with specific subreddit
start-reddit-subreddit:
	@if [ -z "$(SUBREDDIT)" ]; then \
		echo "$(RED)❌ Error: SUBREDDIT variable not set$(NC)"; \
		echo "$(YELLOW)Usage: make start-reddit-subreddit SUBREDDIT=cats$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)🚀 Starting ImageFlasherWGPU (Reddit r/$(SUBREDDIT))...$(NC)"
	@echo "$(YELLOW)🌐 Ikeda interface: http://localhost:$(WEB_PORT)$(NC)"
	@echo "$(YELLOW)📡 WebSocket server will run on port: $(WEBSOCKET_PORT)$(NC)"
	@echo "$(YELLOW)🔍 Source: Reddit r/$(SUBREDDIT)$(NC)"
	@echo "$(YELLOW)⏹️  Press Ctrl+C to stop when ready$(NC)"
	@echo ""
	@node server.js --reddit --subreddit $(SUBREDDIT)

# Clean build artifacts
clean: stop
	@echo "$(YELLOW)🧹 Cleaning build artifacts...$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)✅ Build directory cleaned$(NC)"

# Check server status
status:
	@echo "$(YELLOW)📊 Checking server status...$(NC)"
	@echo "Web server (port $(WEB_PORT)):"
	@lsof -i:$(WEB_PORT) | head -2 || echo "  Not running"
	@echo ""
	@echo "WebSocket server (port $(WEBSOCKET_PORT)):"
	@lsof -i:$(WEBSOCKET_PORT) | head -2 || echo "  Not running"

# Development shortcuts
dev: rebuild
ikeda: rebuild-ikeda
reddit: rebuild-reddit
quick-start: stop start
quick-ikeda: stop start-ikeda
quick-reddit: stop start-reddit
