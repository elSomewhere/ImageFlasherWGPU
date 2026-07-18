# Autonomous crawler and real-time artifact architecture

The crawler is designed as an ephemeral journey, not an archive. Each process starts
a new session, explores in memory, keeps only bounded rolling state, and streams
normalized artifacts to any connected installation viewers.

## Data path

```text
Wikipedia random + external exits ─┐
Wikidata official-site exits ──────┼─> scored, host-stratified URL frontier
operator seeds and keywords ───────┘                 │
                                                     v
                                            8 page workers
                                                     │
                            links admitted immediately│media candidates
                                                     v
                                          bounded priority queue
                                                     │
                                            8 media workers
                                                     │
                                   processor registry + deduplication
                                                     │
                                                     v
                              256-artifact in-memory rolling broker
                                      │                 │
                            viewer A queue       viewer B queue
                                      │                 │
                           16 GPU-credit window per viewer
                                      │
                                      v
                           WASM decode -> 256-layer GPU ring
                                      │
                                      v
                              one instanced tile draw
```

Page traversal, media downloading, decoding, publication, and display are separate
stages. A slow image or viewer cannot stop link discovery. Queue overload sheds old
work locally rather than growing memory without a bound.

## Why collected artifacts now reach the wall

The broker collects whether or not a viewer exists. A new viewer first receives the
current rolling snapshot and then the live stream. It may have only 16 unconfirmed
frames in flight: each credit is returned after the WASM renderer reports a real GPU
upload, or explicitly rejects an unsupported/bad artifact. This prevents a 256-frame
snapshot from overrunning the 32-item decode queues.

The GPU owns one 256-layer, 384 × 384 texture array. A storage buffer maps all visible
tiles to resident layers, so a frame updates tile state once and renders the wall in
one instanced draw instead of creating bind groups and issuing a draw for every tile.

The status bar exposes `received`, `decoded`, `uploaded`, and `presented` counts. A
persistent difference between stages identifies exactly where pressure or rejection
occurs.

## Autonomous path selection

The public `exploration` control is the main behavioral parameter:

- `0`: focused, low-temperature selection weighted toward keyword relevance.
- `1`: high-temperature selection with stronger host freshness, random variation,
  external-domain exits, and teleports.
- Intermediate values blend relevance and wandering; steering biases but never gates.

Selection is host-stratified and the frontier has global and per-registrable-domain
caps. The unattended autopilot watches visual novelty, recent domain entropy, and
failure rate. It reheats a repetitive journey and requests a fresh seed at chapter
boundaries when it appears stuck. The seed rotation spans multilingual Wikipedia
articles, their external links, and Wikidata P856 official-site links. Keyword input
also opens a direct Wikimedia Commons media lane with rights metadata.

Exact SHA-256 deduplication is followed by bounded visual deduplication using dHash
plus a compact color histogram. All archives, seen-URL sets, queues, source histories,
and broker storage are bounded; URL memory also expires after a TTL.

## Compliance and safety invariants

The real-web composition root refuses to start without the compliance transport.
It applies the same policy to pages, media, APIs, redirects, and robots requests:

- Only HTTP(S), credential-free, canonical URLs are accepted.
- DNS answers must all be public addresses; literal private, loopback, link-local,
  multicast, reserved, and unspecified addresses are blocked.
- The connected peer is checked again, and every redirect is revalidated.
- `robots.txt` is cached and checked before each target request. A 4xx robots response
  is treated as unavailable/allow, while network and 5xx failures temporarily deny.
- `Crawl-delay` raises the per-host delay when present.
- One shared scheduler enforces global concurrency, per-origin concurrency, request
  spacing, transient backoff with jitter, and circuit breaking.
- Pages, robots files, images, redirect chains, decoded pixel counts, frontier size,
  queues, and caches all have hard limits.
- `noindex`, `nofollow`, page-level robots headers, action URLs, credentials, common
  crawl traps, and tracking parameters are handled generically.

The default `broad` content policy displays transient public-web material and stores
no crawl payloads on disk. The optional `open-license` policy currently acts as a
known-rights gate: artifacts without explicit rights metadata are rejected at the
broker. It is a provenance control, not a substitute for a legal review of a public
installation.

The installation identifies itself with a descriptive User-Agent. Before deployment,
replace its project URL with a stable page that includes operator contact details.

## Artifact protocol and future media

Every producer uses protocol v1:

```text
uint32 little-endian JSON-header length | UTF-8 JSON header | payload bytes
```

The header carries session and sequence IDs, kind, MIME type, dimensions/duration,
source and page URLs, acquisition time, content hash, producer, novelty/score,
rights/provenance, and extensible metadata. Generated Ikeda frames, the legacy Reddit
source, and the autonomous crawler all publish through the same broker.

Adding text, audio, or video does not require crawler or broker changes:

1. Discover a `MediaCandidate` with a new `kind`.
2. Register an `ArtifactProcessor` for that kind.
3. Emit the normalized artifact through the existing sink.
4. Add a renderer/sonifier consumer and return `skipped` until it is supported.

## Running

```bash
make setup-python
npm install
npm run build:wasm
npm run start:web-crawler
```

Open `http://localhost:8000`. No seed or keyword configuration is required. Examples
of optional steering:

```bash
node server.js --web-crawler \
  --keywords "brutalism,radio astronomy" \
  --seeds "https://en.wikipedia.org/wiki/Brutalist_architecture"

CRAWLER_EXPLORATION=0.85 \
CRAWLER_PAGE_DELAY=1.5 \
CRAWLER_CONTENT_POLICY=open-license \
node server.js --web-crawler
```

Direct Python flags additionally expose page/media worker counts, global/per-origin
concurrency, deterministic random seed, and autopilot selection:

```bash
.venv/bin/python src/python/web_crawler_server.py --help
```

Runtime steering is available in the browser or through the Node control endpoints:

- `GET /api/crawler/state`
- `POST /api/crawler/keywords`
- `POST /api/crawler/seeds`
- `POST /api/crawler/exploration`
- `POST /api/crawler/autopilot`
- `POST /api/crawler/content-policy`

The state response includes frontier/media/broker occupancy, delivery
acknowledgements, source diversity, scheduler circuits, robots cache status,
admission/rejection/dedup counts, and recent bounded events/errors.

## Verification

```bash
npm test
npm run build:wasm
node --check server.js
node --check public/app.js
```

Unit tests cover discovery, traversal, selection, deduplication, protocol framing,
broker eviction and GPU-credit pacing, robots behavior, scheduler cancellation,
URL policy, image limits, and control validation. The end-to-end smoke path should
also confirm cross-origin isolation, WebGPU availability, increasing delivery-stage
counters, no runtime abort/WebGPU error, and a non-uniform canvas pixel sample.
