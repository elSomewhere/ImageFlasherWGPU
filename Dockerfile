# Multi-stage Dockerfile for ImageFlasherWGPU
FROM node:18-alpine as base

# Install system dependencies for Python and compilation
RUN apk add --no-cache python3 py3-pip build-base python3-dev \
    cairo-dev pango-dev jpeg-dev freetype-dev \
    pkgconfig

# Create app directory
WORKDIR /app

# Copy package files
COPY package*.json ./
COPY requirements.txt ./

# Install Node.js dependencies
RUN npm ci --only=production

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build WebAssembly modules if not already built
# Note: In production, you should build these in CI/CD and copy the artifacts
RUN if [ ! -f "public/index.wasm" ]; then \
        echo "Warning: WASM files not found. Building from source..." && \
        apk add --no-cache emscripten && \
        npm run build:wasm || echo "WASM build failed, using pre-built files"; \
    fi

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S appuser -u 1001

# Change ownership of app directory
RUN chown -R appuser:nodejs /app
USER appuser

# Expose ports
EXPOSE 8000 5010

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application in Reddit mode by default
CMD ["node", "server.js", "--reddit", "--subreddit", "worldnews"]