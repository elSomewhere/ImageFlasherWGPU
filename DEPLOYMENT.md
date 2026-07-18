# Installation deployment guide

This project is optimized for a long-running exhibition machine. The crawler and
artifact history are deliberately ephemeral: restarting creates a fresh journey and
no harvested payloads are persisted.

## Build and run

```bash
make setup-python
npm install
npm run build:wasm
npm run start:web-crawler
```

The first WASM build downloads Emscripten's official Dawn WebGPU port. The generated
`public/index.js` and `public/index.wasm` are checked into this project, so a machine
that only runs the installation does not need to rebuild unless the C++ changes.

Default listeners:

| Service | Default | Purpose |
|---|---:|---|
| Node web UI | `0.0.0.0:8000` | Interface, health, and crawler HTTP controls |
| Artifact WebSocket | `127.0.0.1:5010` | Versioned artifact stream |
| Crawler control WebSocket | `127.0.0.1:5011` | Node-to-crawler steering/state |

Open `http://localhost:8000`. The server sets the COOP/COEP headers required for
threaded WebAssembly and exposes `/health` for supervision.

## Environment

```bash
WEB_PORT=8000
WEBSOCKET_PORT=5010
CRAWLER_CONTROL_PORT=5011

# Keep control local. Bind the artifact stream externally only when LAN viewers need it.
CRAWLER_IMAGE_HOST=127.0.0.1
CRAWLER_CONTROL_HOST=127.0.0.1

# Optional behavior/safety tuning
CRAWLER_EXPLORATION=0.55
CRAWLER_CONTENT_POLICY=broad
CRAWLER_PAGE_DELAY=1.0
CRAWLER_GLOBAL_CONCURRENCY=16
```

The browser discovers the configured artifact port from `/api/runtime-config`. It can
also be overridden per viewer with `?imageWs=wss://installation.example/stream`.

## Exhibition service management

Run the Node parent under launchd, systemd, Docker, or another supervisor that:

- starts it from the repository root;
- uses `node server.js --web-crawler`;
- restarts only on unexpected exit, with a delay to avoid a crash loop;
- sends SIGTERM for shutdown so the Python child is stopped;
- captures stdout/stderr and rotates logs;
- health-checks `GET /health`.

Do not run separate replicas against one browser stream without an explicit upstream
broker. Each process is intentionally its own journey and has its own in-memory ring.

## LAN and reverse proxy

For a trusted local network, bind the artifact socket on all interfaces:

```bash
CRAWLER_IMAGE_HOST=0.0.0.0 node server.js --web-crawler
```

Keep port 5011 private. If the UI is served over HTTPS, the browser requires a secure
`wss://` artifact endpoint. Terminate TLS in the reverse proxy, forward that endpoint
to `127.0.0.1:5010`, and provide its URL through the `imageWs` query parameter (or
adapt `/api/runtime-config` for the deployment). Preserve WebSocket upgrade headers.

The proxy must retain these response headers on HTML, JavaScript, and WASM:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## Operational checks

Before opening the installation:

```bash
npm test
npm run build:wasm
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/crawler/state
```

In the UI, verify:

- connection status is `CONNECTED`, not `GPU ERROR` or `WASM ERROR`;
- received, decoded, uploaded, and presented counts continue increasing;
- the uploaded count trails received by no more than the normal 16-frame credit window;
- the crawler log changes domains over time and the frontier/media queues remain bounded;
- WebGPU is enabled in the exhibition browser.

The state endpoint reports open origin circuits, robots cache state, rejections,
duplicates, queue occupancy, client drops, and delivery acknowledgements. Alert on a
counter that stops moving, persistent open circuits across many origins, or a growing
gap between delivery stages.

## Data and compliance

- No crawler database or harvested media directory is created.
- The broker, dedup archives, frontier, seen sets, and event logs are bounded memory.
- The real-web entry point cannot disable robots/SSRF/politeness layers.
- Replace the default User-Agent project URL with a stable operator/contact page.
- Review the content policy, local law, venue requirements, and expected audience
  before a public deployment. `open-license` is a metadata gate, not legal advice.

See [CRAWLER_ARCHITECTURE.md](CRAWLER_ARCHITECTURE.md) for detailed invariants and
extension points.
