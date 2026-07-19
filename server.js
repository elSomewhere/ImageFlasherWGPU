#!/usr/bin/env node

/**
 * ImageFlasherWGPU Node.js Server
 * ===============================
 * 
 * Unified server that handles both the web server and Python image streaming.
 * Properly integrates all components into a single Node.js project.
 */

const express = require('express');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const WebSocket = require('ws');

// Configuration
const WEB_PORT = Number(process.env.WEB_PORT || 8000);
const WEBSOCKET_PORT = Number(process.env.WEBSOCKET_PORT || 5010);
const CRAWLER_CONTROL_PORT = Number(process.env.CRAWLER_CONTROL_PORT || 5011);
const CRAWLER_IMAGE_HOST = process.env.CRAWLER_IMAGE_HOST || '127.0.0.1';
const CRAWLER_CONTROL_HOST = process.env.CRAWLER_CONTROL_HOST || '127.0.0.1';
const CRAWLER_EXPLORATION = process.env.CRAWLER_EXPLORATION || '0.55';
const CRAWLER_CONTENT_POLICY = process.env.CRAWLER_CONTENT_POLICY || 'broad';
const CRAWLER_PAGE_DELAY = process.env.CRAWLER_PAGE_DELAY || '1.0';
const CRAWLER_GLOBAL_CONCURRENCY = process.env.CRAWLER_GLOBAL_CONCURRENCY || '16';

class ImageFlasherServer {
    constructor() {
        this.webServer = null;
        this.pythonProcess = null;
        this.running = false;
        this.mode = 'ikeda';
        this.crawlerControlHost = CRAWLER_CONTROL_HOST;
    }

    async start(options = {}) {
        const mode = options.mode || 'ikeda';
        const subreddit = options.subreddit || 'worldnews';
        this.mode = mode;
        this.crawlerControlHost = options.crawlerControlHost || CRAWLER_CONTROL_HOST;
        console.log('🚀 Starting ImageFlasherWGPU Server');
        console.log('====================================');

        try {
            // Start web server
            await this.startWebServer();
            
            // Start Python image server
            await this.startImageServer(options);
            
            this.running = true;
            this.printStatus(mode, subreddit);
            
            // Handle graceful shutdown
            this.setupShutdownHandlers();
            
        } catch (error) {
            console.error('❌ Failed to start server:', error);
            process.exit(1);
        }
    }

    sendCrawlerCommand(command) {
        return new Promise((resolve, reject) => {
            const ws = new WebSocket(`ws://${this.crawlerControlHost}:${CRAWLER_CONTROL_PORT}`);
            const timeout = setTimeout(() => {
                ws.terminate();
                reject(new Error('Crawler control request timed out'));
            }, 5000);

            ws.on('open', () => {
                ws.send(JSON.stringify(command));
            });

            ws.on('message', (data) => {
                clearTimeout(timeout);
                try {
                    resolve(JSON.parse(data.toString()));
                } catch (error) {
                    reject(error);
                } finally {
                    ws.close();
                }
            });

            ws.on('error', (error) => {
                clearTimeout(timeout);
                reject(error);
            });
        });
    }

    startWebServer() {
        return new Promise((resolve, reject) => {
            const app = express();
            app.use(express.json({ limit: '64kb' }));

            // CORS headers for WebGPU - MUST be set before static files
            app.use((req, res, next) => {
                res.header('Cross-Origin-Opener-Policy', 'same-origin');
                res.header('Cross-Origin-Embedder-Policy', 'require-corp');
                res.header('Cross-Origin-Resource-Policy', 'cross-origin');
                next();
            });

            // Serve static files from public directory
            app.use(express.static('public', {
                setHeaders: (res, path) => {
                    // Ensure WASM files have proper headers
                    if (path.endsWith('.wasm')) {
                        res.header('Content-Type', 'application/wasm');
                    }
                    if (path.endsWith('.js')) {
                        res.header('Content-Type', 'application/javascript');
                    }
                }
            }));

            // Main route - serve the Ikeda interface
            app.get('/', (req, res) => {
                // Explicitly set cross-origin isolation headers
                res.header('Cross-Origin-Opener-Policy', 'same-origin');
                res.header('Cross-Origin-Embedder-Policy', 'require-corp');
                res.header('Cross-Origin-Resource-Policy', 'cross-origin');
                res.sendFile(path.join(__dirname, 'public', 'index.html'));
            });

            // Health check endpoint
            app.get('/health', (req, res) => {
                res.json({ 
                    status: 'running',
                    web_port: WEB_PORT,
                    websocket_port: WEBSOCKET_PORT,
                    timestamp: new Date().toISOString()
                });
            });

            app.get('/api/runtime-config', (req, res) => {
                res.json({
                    websocket_port: WEBSOCKET_PORT,
                    crawler_control_port: CRAWLER_CONTROL_PORT,
                    crawler_enabled: this.mode === 'web-crawler',
                    mode: this.mode
                });
            });

            app.post('/api/crawler/keywords', async (req, res) => {
                try {
                    const keywords = Array.isArray(req.body?.keywords)
                        ? req.body.keywords
                        : String(req.body?.keywords || '').split(',');
                    const response = await this.sendCrawlerCommand({
                        type: 'set_keywords',
                        keywords
                    });
                    res.json(response);
                } catch (error) {
                    res.status(503).json({ ok: false, error: error.message });
                }
            });

            app.post('/api/crawler/seeds', async (req, res) => {
                try {
                    const seeds = Array.isArray(req.body?.seeds)
                        ? req.body.seeds
                        : String(req.body?.seeds || '').split(/\s+/);
                    const response = await this.sendCrawlerCommand({
                        type: 'add_seeds',
                        seeds
                    });
                    res.json(response);
                } catch (error) {
                    res.status(503).json({ ok: false, error: error.message });
                }
            });

            app.post('/api/crawler/exploration', async (req, res) => {
                try {
                    const response = await this.sendCrawlerCommand({
                        type: 'set_exploration',
                        exploration: Number(req.body?.exploration)
                    });
                    res.json(response);
                } catch (error) {
                    res.status(503).json({ ok: false, error: error.message });
                }
            });

            app.post('/api/crawler/autopilot', async (req, res) => {
                try {
                    const response = await this.sendCrawlerCommand({
                        type: 'set_autopilot',
                        enabled: Boolean(req.body?.enabled)
                    });
                    res.json(response);
                } catch (error) {
                    res.status(503).json({ ok: false, error: error.message });
                }
            });

            app.post('/api/crawler/content-policy', async (req, res) => {
                try {
                    const response = await this.sendCrawlerCommand({
                        type: 'set_content_policy',
                        policy: String(req.body?.policy || 'broad')
                    });
                    res.json(response);
                } catch (error) {
                    res.status(503).json({ ok: false, error: error.message });
                }
            });

            app.get('/api/crawler/state', async (req, res) => {
                try {
                    const response = await this.sendCrawlerCommand({ type: 'get_state' });
                    res.json(response);
                } catch (error) {
                    res.status(503).json({ ok: false, error: error.message });
                }
            });

            // Start the server
            this.webServer = app.listen(WEB_PORT, () => {
                console.log('✅ Web server started successfully');
                console.log(`🌐 Interface: http://localhost:${WEB_PORT}`);
                resolve();
            });

            this.webServer.on('error', (error) => {
                console.error('❌ Web server error:', error);
                reject(error);
            });
        });
    }

    startImageServer(options = {}) {
        return new Promise((resolve, reject) => {
            const mode = options.mode || 'ikeda';
            const subreddit = options.subreddit || 'worldnews';
            const crawlerImageHost = options.crawlerImageHost || CRAWLER_IMAGE_HOST;
            const crawlerControlHost = options.crawlerControlHost || CRAWLER_CONTROL_HOST;
            let scriptPath;
            let args = [];
            // Resolve Python interpreter: prefer project's .venv if available
            const projectRoot = __dirname;
            const venvPython = path.join(projectRoot, '.venv', 'bin', 'python');
            const pythonCmd = fs.existsSync(venvPython) ? venvPython : 'python3';

            // Determine which Python script to run
            switch (mode) {
                case 'ikeda':
                case 'generated':
                    scriptPath = path.join(__dirname, 'src', 'python', 'ImageCreator_Ikeda.py');
                    args = ['--host', crawlerImageHost, '--port', String(WEBSOCKET_PORT)];
                    console.log('🎨 Starting Ikeda data visualization server...');
                    break;
                case 'reddit':
                    scriptPath = path.join(__dirname, 'src', 'python', 'scraper_3.py');
                    args = [
                        '--subreddit', subreddit,
                        '--host', crawlerImageHost,
                        '--port', String(WEBSOCKET_PORT)
                    ];
                    console.log(`🔍 Starting Reddit crawler (r/${subreddit})...`);
                    break;
                case 'web-crawler':
                    scriptPath = path.join(__dirname, 'src', 'python', 'web_crawler_server.py');
                    args = [
                        '--image-host', crawlerImageHost,
                        '--image-port', String(WEBSOCKET_PORT),
                        '--control-host', crawlerControlHost,
                        '--control-port', String(CRAWLER_CONTROL_PORT),
                        '--exploration', CRAWLER_EXPLORATION,
                        '--content-policy', CRAWLER_CONTENT_POLICY,
                        '--page-delay', CRAWLER_PAGE_DELAY,
                        '--global-concurrency', CRAWLER_GLOBAL_CONCURRENCY
                    ];
                    for (const keyword of options.keywords || []) {
                        args.push('--keyword', keyword);
                    }
                    for (const seed of options.seeds || []) {
                        args.push('--seed', seed);
                    }
                    for (const plugin of options.seedPlugins || []) {
                        args.push('--seed-plugin', plugin);
                    }
                    if (options.maxDepth !== null && options.maxDepth !== undefined) {
                        args.push('--max-depth', String(options.maxDepth));
                    }
                    if (options.enableCommons) {
                        args.push('--enable-commons');
                    }
                    console.log('🔎 Starting generic topic-steered web crawler...');
                    break;
                default:
                    scriptPath = path.join(__dirname, 'src', 'python', 'ImageCreator_Ikeda.py');
                    args = ['--host', crawlerImageHost, '--port', String(WEBSOCKET_PORT)];
                    console.log('🎨 Starting default Ikeda server...');
            }

            // Check if Python script exists
            if (!fs.existsSync(scriptPath)) {
                reject(new Error(`Python script not found: ${scriptPath}`));
                return;
            }

            // Start Python process
            this.pythonProcess = spawn(pythonCmd, [scriptPath, ...args], {
                stdio: ['pipe', 'pipe', 'pipe'],
                cwd: __dirname
            });

            // Handle Python process output
            this.pythonProcess.stdout.on('data', (data) => {
                console.log(`[Python] ${data.toString().trim()}`);
            });

            this.pythonProcess.stderr.on('data', (data) => {
                console.error(`[Python Error] ${data.toString().trim()}`);
            });

            let exitedBeforeStartup = false;
            this.pythonProcess.on('close', (code) => {
                exitedBeforeStartup = true;
                if (code !== 0 && this.running) {
                    console.error(`❌ Python process exited with code ${code}`);
                } else {
                    console.log('🛑 Python process stopped');
                }
            });

            this.pythonProcess.on('error', (error) => {
                console.error('❌ Failed to start Python process:', error);
                reject(error);
            });

            const waitForProcess = () => {
                setTimeout(() => {
                    if (exitedBeforeStartup || !this.pythonProcess || this.pythonProcess.killed) {
                        reject(new Error('Python process failed to start'));
                    } else {
                        console.log('✅ Image server started successfully');
                        console.log(`📡 WebSocket server on port: ${WEBSOCKET_PORT}`);
                        resolve();
                    }
                }, 2000);
            };

            const waitForCrawlerControl = async () => {
                const deadline = Date.now() + 8000;
                while (Date.now() < deadline) {
                    if (exitedBeforeStartup || !this.pythonProcess || this.pythonProcess.killed) {
                        reject(new Error('Python crawler process exited before startup completed'));
                        return;
                    }
                    try {
                        const response = await this.sendCrawlerCommand({ type: 'get_state' });
                        if (response && response.ok) {
                            console.log('✅ Image server started successfully');
                            console.log(`📡 WebSocket server on port: ${WEBSOCKET_PORT}`);
                            resolve();
                            return;
                        }
                    } catch (error) {
                        await new Promise((r) => setTimeout(r, 250));
                    }
                }
                reject(new Error('Python crawler control service did not respond'));
            };

            if (mode === 'web-crawler') {
                waitForCrawlerControl();
            } else {
                waitForProcess();
            }
        });
    }

    printStatus(mode, subreddit) {
        console.log('\n' + '='.repeat(50));
        console.log('🎉 ImageFlasherWGPU is running!');
        console.log('='.repeat(50));
        console.log(`🌐 Web Interface: http://localhost:${WEB_PORT}`);
        console.log(`📡 WebSocket Server: ws://localhost:${WEBSOCKET_PORT}`);
        console.log(`🎨 Image Mode: ${mode.toUpperCase()}`);
        
        if (mode === 'reddit') {
            console.log(`📱 Subreddit: r/${subreddit}`);
        } else if (mode === 'web-crawler') {
            console.log(`🧭 Crawler Control API: http://localhost:${WEB_PORT}/api/crawler/state`);
            console.log(`🎛️  Crawler Control WS: ws://${this.crawlerControlHost}:${CRAWLER_CONTROL_PORT}`);
        }
        
        console.log('\n💡 Features:');
        console.log('   • Real-time WebGPU rendering');
        console.log('   • Ikeda-inspired data visualization');
        console.log('   • 13 distinct visual modes');
        console.log('   • Live keyboard controls');
        console.log('   • Data analysis overlay');
        
        console.log('\n🎹 Keyboard Shortcuts:');
        console.log('   • 1-9, 0, -, = : Switch modes');
        console.log('   • B/G/D/F/S/M/P/N : Quick mode access');
        console.log('   • T : Cycle threshold');
        console.log('   • R : Reset defaults');
        console.log('   • ESC : Toggle fullscreen');
        
        console.log('\n⏹️  Press Ctrl+C to stop');
        console.log('='.repeat(50));
    }

    setupShutdownHandlers() {
        const shutdown = () => {
            console.log('\n🛑 Shutting down server...');
            this.running = false;
            
            if (this.pythonProcess) {
                this.pythonProcess.kill('SIGTERM');
            }
            
            if (this.webServer) {
                this.webServer.close(() => {
                    console.log('✅ Web server stopped');
                    process.exit(0);
                });
            } else {
                process.exit(0);
            }
        };

        process.on('SIGINT', shutdown);
        process.on('SIGTERM', shutdown);
        process.on('uncaughtException', (error) => {
            console.error('❌ Uncaught exception:', error);
            shutdown();
        });
    }
}

// Command line argument parsing
function parseArgs() {
    const args = process.argv.slice(2);
    let mode = 'ikeda';
    let subreddit = 'worldnews';
    let keywords = [];
    let seeds = [];
    let seedPlugins = [];
    let maxDepth = null;
    let enableCommons = false;
    let crawlerImageHost = CRAWLER_IMAGE_HOST;
    let crawlerControlHost = CRAWLER_CONTROL_HOST;

    for (let i = 0; i < args.length; i++) {
        switch (args[i]) {
            case '--ikeda':
                mode = 'ikeda';
                break;
            case '--generated':
                mode = 'generated';
                break;
            case '--reddit':
                mode = 'reddit';
                break;
            case '--web-crawler':
                mode = 'web-crawler';
                break;
            case '--keyword':
            case '--keywords':
                if (i + 1 < args.length) {
                    keywords = args[i + 1].split(',').map((value) => value.trim()).filter(Boolean);
                    i++;
                }
                break;
            case '--seed':
            case '--seeds':
                if (i + 1 < args.length) {
                    seeds = args[i + 1].split(',').map((value) => value.trim()).filter(Boolean);
                    i++;
                }
                break;
            case '--seed-plugin':
            case '--seed-plugins':
                if (i + 1 < args.length) {
                    seedPlugins = args[i + 1].split(',').map((value) => value.trim()).filter(Boolean);
                    i++;
                }
                break;
            case '--max-depth':
                if (i + 1 < args.length) {
                    maxDepth = args[i + 1];
                    i++;
                }
                break;
            case '--enable-commons':
                enableCommons = true;
                break;
            case '--crawler-image-host':
                if (i + 1 < args.length) {
                    crawlerImageHost = args[i + 1];
                    i++;
                }
                break;
            case '--crawler-control-host':
                if (i + 1 < args.length) {
                    crawlerControlHost = args[i + 1];
                    i++;
                }
                break;
            case '--subreddit':
                if (i + 1 < args.length) {
                    subreddit = args[i + 1];
                    i++; // Skip next argument
                }
                break;
            case '--help':
            case '-h':
                console.log(`
ImageFlasherWGPU Server

Usage: node server.js [options]

Options:
  --ikeda              Use Ikeda data visualization mode (default)
  --generated          Use generated images mode  
  --reddit             Use Reddit scraper mode
  --web-crawler        Use generic topic-steered web crawler mode
  --subreddit <name>   Specify subreddit for Reddit mode (default: worldnews)
  --keywords <terms>   Comma-separated crawler keywords (steer scoring)
  --seeds <urls>       Comma-separated crawler seed URLs (the walk's entry points)
  --seed-plugins <names> Comma-separated opt-in seed plugins
                         (wikipedia_random, wikidata_official_sites)
  --max-depth <n>      Cap link depth from seeds (default: unlimited)
  --enable-commons     Enable the keyword -> Wikimedia Commons media lane
  --crawler-image-host <host>   Image WebSocket bind host (default: ${CRAWLER_IMAGE_HOST})
  --crawler-control-host <host> Control WebSocket bind host (default: ${CRAWLER_CONTROL_HOST})
  --help, -h           Show this help message

Examples:
  node server.js                           # Ikeda mode (default)
  node server.js --reddit --subreddit cats # Reddit cats images
  node server.js --web-crawler --keywords cats --seeds https://example.com
  node server.js --generated               # Generated images
                `);
                process.exit(0);
                break;
        }
    }

    return {
        mode,
        subreddit,
        keywords,
        seeds,
        seedPlugins,
        maxDepth,
        enableCommons,
        crawlerImageHost,
        crawlerControlHost
    };
}

// Main execution
if (require.main === module) {
    const options = parseArgs();
    const server = new ImageFlasherServer();
    server.start(options);
}

module.exports = ImageFlasherServer;
