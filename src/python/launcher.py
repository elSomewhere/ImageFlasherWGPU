#!/usr/bin/env python3
"""
ImageFlasherWGPU Launcher
=========================

A single script to launch the complete ImageFlasherWGPU application.
Starts both the web server and image streaming server with options for different image sources.

Usage:
    python3 launcher.py                    # Default: generated images
    python3 launcher.py --generated        # Generated VHS-style images
    python3 launcher.py --reddit           # Reddit scraped images
    python3 launcher.py --reddit --subreddit cats  # Custom subreddit
"""

import argparse
import subprocess
import sys
import time
import os
import signal
import threading
from pathlib import Path


class ApplicationLauncher:
    def __init__(self):
        self.web_server_process = None
        self.image_server_process = None
        self.running = True
        
    def start_web_server(self):
        """Start the web server in the cmake-build-emscripten directory."""
        build_dir = Path("cmake-build-emscripten")
        if not build_dir.exists():
            print("❌ Error: cmake-build-emscripten directory not found!")
            print("   Please build the project first with: cmake --build cmake-build-emscripten")
            return False
            
        print("🌐 Starting web server on http://localhost:8000...")
        
        try:
            self.web_server_process = subprocess.Popen(
                [sys.executable, "serve.py"],
                cwd=build_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # Give it a moment to start
            time.sleep(1)
            if self.web_server_process.poll() is None:
                print("✅ Web server started successfully")
                return True
            else:
                print("❌ Web server failed to start")
                return False
        except Exception as e:
            print(f"❌ Error starting web server: {e}")
            return False
    
    def start_image_server_generated(self, ikeda_mode=False):
        """Start the generated images server."""
        if ikeda_mode:
            print("🎨 Starting Ikeda data visualization server...")
            script_name = "ImageCreator_Ikeda.py"
        else:
            print("🎨 Starting generated images server...")
            script_name = "ImageCreator.py"
        
        try:
            self.image_server_process = subprocess.Popen(
                [sys.executable, script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            time.sleep(1)
            if self.image_server_process.poll() is None:
                if ikeda_mode:
                    print("✅ Ikeda data visualization server started successfully")
                else:
                    print("✅ Generated images server started successfully")
                return True
            else:
                print("❌ Image server failed to start")
                return False
        except Exception as e:
            print(f"❌ Error starting image server: {e}")
            return False
    
    def start_image_server_reddit(self, subreddit="worldnews"):
        """Start the Reddit scraper server."""
        print(f"🔍 Starting Reddit scraper for r/{subreddit}...")
        
        # Create a temporary modified scraper script for custom subreddit
        if subreddit != "worldnews":
            self.create_custom_scraper(subreddit)
            script_name = "scraper_custom.py"
        else:
            script_name = "scraper_3.py"
            
        try:
            self.image_server_process = subprocess.Popen(
                [sys.executable, script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            time.sleep(1)
            if self.image_server_process.poll() is None:
                print("✅ Reddit scraper started successfully")
                return True
            else:
                print("❌ Reddit scraper failed to start")
                return False
        except Exception as e:
            print(f"❌ Error starting Reddit scraper: {e}")
            return False
    
    def create_custom_scraper(self, subreddit):
        """Create a custom scraper script for the specified subreddit."""
        with open("scraper_3.py", "r") as f:
            content = f.read()
        
        # Replace the subreddit in the scraping thread function
        modified_content = content.replace(
            'subreddit="worldnews"',
            f'subreddit="{subreddit}"'
        )
        
        with open("scraper_custom.py", "w") as f:
            f.write(modified_content)
    
    def cleanup_custom_scraper(self):
        """Remove the temporary custom scraper script."""
        custom_script = Path("scraper_custom.py")
        if custom_script.exists():
            custom_script.unlink()
    
    def monitor_processes(self):
        """Monitor both processes and handle output."""
        def monitor_web_server():
            if self.web_server_process:
                for line in iter(self.web_server_process.stdout.readline, ''):
                    if not self.running:
                        break
                    if line.strip():
                        print(f"[WEB] {line.strip()}")
        
        def monitor_image_server():
            if self.image_server_process:
                for line in iter(self.image_server_process.stdout.readline, ''):
                    if not self.running:
                        break
                    if line.strip():
                        print(f"[IMG] {line.strip()}")
        
        # Start monitoring threads
        if self.web_server_process:
            threading.Thread(target=monitor_web_server, daemon=True).start()
        if self.image_server_process:
            threading.Thread(target=monitor_image_server, daemon=True).start()
    
    def stop_servers(self):
        """Stop both servers gracefully."""
        print("\n🛑 Stopping servers...")
        self.running = False
        
        if self.image_server_process:
            try:
                self.image_server_process.terminate()
                self.image_server_process.wait(timeout=5)
                print("✅ Image server stopped")
            except subprocess.TimeoutExpired:
                self.image_server_process.kill()
                print("⚠️  Image server force killed")
            except Exception as e:
                print(f"❌ Error stopping image server: {e}")
        
        if self.web_server_process:
            try:
                self.web_server_process.terminate()
                self.web_server_process.wait(timeout=5)
                print("✅ Web server stopped")
            except subprocess.TimeoutExpired:
                self.web_server_process.kill()
                print("⚠️  Web server force killed")
            except Exception as e:
                print(f"❌ Error stopping web server: {e}")
        
        self.cleanup_custom_scraper()
    
    def run(self, mode="generated", subreddit="worldnews", ikeda_mode=False):
        """Run the complete application."""
        print("🚀 Starting ImageFlasherWGPU Application")
        print("=" * 50)
        
        # Start web server
        if not self.start_web_server():
            return False
        
        # Start appropriate image server
        if mode == "generated":
            if not self.start_image_server_generated(ikeda_mode):
                self.stop_servers()
                return False
        elif mode == "reddit":
            if not self.start_image_server_reddit(subreddit):
                self.stop_servers()
                return False
        
        # Start monitoring
        self.monitor_processes()
        
        print("\n" + "=" * 50)
        print("🎉 Application started successfully!")
        print(f"🌐 Web interface: http://localhost:8000")
        if ikeda_mode:
            print(f"🎨 Image mode: IKEDA DATA VISUALIZATION")
            print(f"🔬 Analysis: Real-time image data extraction")
            print(f"⚫ Aesthetic: Pure black & white minimalism")
        else:
            print(f"🎨 Image mode: {mode.upper()}")
        if mode == "reddit":
            print(f"📱 Subreddit: r/{subreddit}")
        print("\n💡 Tip: Use the control panel in the web interface to adjust effects")
        if ikeda_mode:
            print("🎹 Keyboard shortcuts: 1-4 (modes), B (black/white), G (grid), T (threshold), ESC (fullscreen)")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 50)
        
        # Wait for interrupt
        try:
            while self.running:
                time.sleep(1)
                # Check if processes are still running
                if self.web_server_process and self.web_server_process.poll() is not None:
                    print("❌ Web server stopped unexpectedly")
                    break
                if self.image_server_process and self.image_server_process.poll() is not None:
                    print("❌ Image server stopped unexpectedly")
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_servers()
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Launch the ImageFlasherWGPU application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 launcher.py                           # Generated images (default)
  python3 launcher.py --generated               # Generated VHS-style images  
  python3 launcher.py --ikeda                   # Ikeda data visualization mode
  python3 launcher.py --reddit                  # Reddit worldnews images
  python3 launcher.py --reddit --subreddit cats # Reddit cats images
  python3 launcher.py --reddit --subreddit art  # Reddit art images
        """
    )
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--generated", 
        action="store_true", 
        help="Use generated VHS-style images (default)"
    )
    mode_group.add_argument(
        "--ikeda", 
        action="store_true", 
        help="Use Ikeda-inspired data visualization mode (black & white, data-driven)"
    )
    mode_group.add_argument(
        "--reddit", 
        action="store_true", 
        help="Use Reddit scraped images"
    )
    
    parser.add_argument(
        "--subreddit", 
        default="worldnews",
        help="Subreddit to scrape (only with --reddit, default: worldnews)"
    )
    
    args = parser.parse_args()
    
    # Determine mode and ikeda flag
    ikeda_mode = args.ikeda
    if args.reddit:
        mode = "reddit"
    else:
        mode = "generated"  # Default
    
    # Check dependencies
    try:
        import requests
        import websockets
        from PIL import Image
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("📦 Please install requirements: pip3 install -r requirements.txt")
        return 1
    
    # Setup signal handlers
    launcher = ApplicationLauncher()
    
    def signal_handler(signum, frame):
        launcher.stop_servers()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the application
    success = launcher.run(mode=mode, subreddit=args.subreddit, ikeda_mode=ikeda_mode)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 