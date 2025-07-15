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

// Configuration
const WEB_PORT = 8000;
const WEBSOCKET_PORT = 5010;

class ImageFlasherServer {
    constructor() {
        this.webServer = null;
        this.pythonProcess = null;
        this.running = false;
    }

    async start(mode = 'ikeda', subreddit = 'worldnews') {
        console.log('🚀 Starting ImageFlasherWGPU Server');
        console.log('====================================');

        try {
            // Start web server
            await this.startWebServer();
            
            // Start Python image server
            await this.startImageServer(mode, subreddit);
            
            this.running = true;
            this.printStatus(mode, subreddit);
            
            // Handle graceful shutdown
            this.setupShutdownHandlers();
            
        } catch (error) {
            console.error('❌ Failed to start server:', error);
            process.exit(1);
        }
    }

    startWebServer() {
        return new Promise((resolve, reject) => {
            const app = express();

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

    startImageServer(mode, subreddit) {
        return new Promise((resolve, reject) => {
            let scriptPath;
            let args = [];

            // Determine which Python script to run
            switch (mode) {
                case 'ikeda':
                case 'generated':
                    scriptPath = path.join(__dirname, 'src', 'python', 'ImageCreator_Ikeda.py');
                    console.log('🎨 Starting Ikeda data visualization server...');
                    break;
                case 'reddit':
                    scriptPath = path.join(__dirname, 'src', 'python', 'scraper_3.py');
                    args = ['--subreddit', subreddit];
                    console.log(`🔍 Starting Reddit crawler (r/${subreddit})...`);
                    break;
                default:
                    scriptPath = path.join(__dirname, 'src', 'python', 'ImageCreator_Ikeda.py');
                    console.log('🎨 Starting default Ikeda server...');
            }

            // Check if Python script exists
            if (!fs.existsSync(scriptPath)) {
                reject(new Error(`Python script not found: ${scriptPath}`));
                return;
            }

            // Start Python process
            this.pythonProcess = spawn('python3', [scriptPath, ...args], {
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

            this.pythonProcess.on('close', (code) => {
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

            // Give Python process time to start
            setTimeout(() => {
                if (this.pythonProcess && !this.pythonProcess.killed) {
                    console.log('✅ Image server started successfully');
                    console.log(`📡 WebSocket server on port: ${WEBSOCKET_PORT}`);
                    resolve();
                } else {
                    reject(new Error('Python process failed to start'));
                }
            }, 2000);
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
  --subreddit <name>   Specify subreddit for Reddit mode (default: worldnews)
  --help, -h           Show this help message

Examples:
  node server.js                           # Ikeda mode (default)
  node server.js --reddit --subreddit cats # Reddit cats images
  node server.js --generated               # Generated images
                `);
                process.exit(0);
                break;
        }
    }

    return { mode, subreddit };
}

// Main execution
if (require.main === module) {
    const { mode, subreddit } = parseArgs();
    const server = new ImageFlasherServer();
    server.start(mode, subreddit);
}

module.exports = ImageFlasherServer; 