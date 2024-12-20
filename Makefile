# Makefile

# Directory where your build artifacts are located
BUILD_DIR = cmake-build-emscripten

# The Python server script
SERVER_SCRIPT = serve.py

.PHONY: serve
serve:
	cd $(BUILD_DIR) && python3 $(SERVER_SCRIPT)
