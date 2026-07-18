#!/usr/bin/env python3
"""
Enhanced ImageCreator with Ikeda-inspired data analysis
Generates images with embedded analytical data for visualization
"""

import asyncio
import argparse
import io
import time
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import logging

from crawler.adapters.sinks.websocket import ArtifactBroker
from crawler.core.types import Artifact, RightsMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== IMAGE ANALYSIS FUNCTIONS ====================

def analyze_image_data(image_array):
    """
    Extract Ryoji Ikeda-inspired data from image
    Returns structured data for visualization
    """
    # Convert to grayscale for analysis
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    analysis = {}
    
    # Basic statistics
    analysis['mean_luminance'] = float(np.mean(gray))
    analysis['variance'] = float(np.var(gray))
    analysis['std_dev'] = float(np.std(gray))
    analysis['min_value'] = int(np.min(gray))
    analysis['max_value'] = int(np.max(gray))
    
    # Histogram analysis
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    analysis['histogram'] = hist.tolist()
    
    # Calculate entropy (information content)
    hist_normalized = hist / np.sum(hist)
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-7))
    analysis['entropy'] = float(entropy)
    
    # Edge detection for complexity measure
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    analysis['edge_density'] = float(edge_density)
    
    # Frequency domain analysis
    fft = np.fft.fft2(gray)
    fft_magnitude = np.abs(fft)
    fft_energy = np.sum(fft_magnitude ** 2)
    analysis['frequency_energy'] = float(fft_energy)
    
    # Find dominant frequencies (simplified)
    fft_shifted = np.fft.fftshift(fft_magnitude)
    center = np.array(fft_shifted.shape) // 2
    high_freq_mask = np.zeros_like(fft_shifted)
    high_freq_mask[center[0]-10:center[0]+10, center[1]-10:center[1]+10] = 1
    high_freq_energy = np.sum(fft_shifted * (1 - high_freq_mask))
    low_freq_energy = np.sum(fft_shifted * high_freq_mask)
    analysis['high_freq_ratio'] = float(high_freq_energy / (low_freq_energy + 1e-7))
    
    # Texture analysis using Local Binary Patterns (simplified)
    def local_binary_pattern_simple(img):
        """Simplified LBP for texture analysis"""
        lbp = np.zeros_like(img)
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                center = img[i, j]
                binary_string = ''
                neighbors = [
                    img[i-1, j-1], img[i-1, j], img[i-1, j+1],
                    img[i, j+1], img[i+1, j+1], img[i+1, j],
                    img[i+1, j-1], img[i, j-1]
                ]
                for neighbor in neighbors:
                    binary_string += '1' if neighbor >= center else '0'
                lbp[i, j] = int(binary_string, 2)
        return lbp
    
    # Sample a smaller region for LBP to improve performance
    sample_region = gray[::4, ::4]  # Downsample for performance
    lbp = local_binary_pattern_simple(sample_region)
    lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    lbp_uniformity = np.sum((lbp_hist / np.sum(lbp_hist)) ** 2)
    analysis['texture_uniformity'] = float(lbp_uniformity)
    
    # Calculate image complexity metrics
    analysis['dynamic_range'] = analysis['max_value'] - analysis['min_value']
    analysis['contrast_ratio'] = analysis['std_dev'] / (analysis['mean_luminance'] + 1)
    
    # Compression estimation (entropy-based)
    analysis['estimated_compression'] = entropy / 8.0  # Rough estimate
    
    # Generate timestamp and metadata
    analysis['timestamp'] = time.time()
    analysis['image_size'] = [gray.shape[1], gray.shape[0]]  # width, height
    
    return analysis

def create_data_visualization_image(width=512, height=512, analysis_data=None):
    """
    Create a purely data-driven image based on analysis
    Returns both the image and its analysis data
    """
    # Create base image
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    if analysis_data is None:
        # Generate random data for visualization
        analysis_data = {
            'mean_luminance': random.uniform(0, 255),
            'variance': random.uniform(0, 5000),
            'entropy': random.uniform(0, 8),
            'edge_density': random.uniform(0, 0.3),
            'high_freq_ratio': random.uniform(0, 2),
            'texture_uniformity': random.uniform(0, 1)
        }
    
    # Data-driven pattern generation
    
    # 1. Grid based on entropy
    grid_size = int(8 + analysis_data.get('entropy', 4) * 8)
    for x in range(0, width, grid_size):
        for y in range(0, height, grid_size):
            # Cell brightness based on local data
            brightness = int(analysis_data.get('mean_luminance', 127))
            if random.random() < analysis_data.get('edge_density', 0.1):
                brightness = 255  # Highlight edges
            
            cell_color = (brightness, brightness, brightness)
            draw.rectangle([x, y, x+grid_size-1, y+grid_size-1], fill=cell_color)
    
    # 2. Barcode patterns based on variance
    variance = analysis_data.get('variance', 1000)
    bar_count = int(variance / 50) + 10
    bar_width = width // bar_count
    
    for i in range(bar_count):
        x = i * bar_width
        # Bar height based on frequency content
        bar_height = int((analysis_data.get('high_freq_ratio', 0.5) * height / 2))
        if i % 2 == 0:  # Alternate pattern
            draw.rectangle([x, 0, x+bar_width//2, bar_height], fill=(255, 255, 255))
    
    # 3. Texture pattern overlay
    texture_density = analysis_data.get('texture_uniformity', 0.5)
    for _ in range(int(texture_density * 1000)):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        draw.point((x, y), fill=(255, 255, 255))
    
    # 4. Data stream along edges
    # Top edge: entropy visualization
    entropy = analysis_data.get('entropy', 4)
    segment_width = width // 64
    for i in range(64):
        x = i * segment_width
        if random.random() < entropy / 8:
            draw.rectangle([x, 0, x+segment_width//2, 4], fill=(255, 255, 255))
    
    # Convert to analysis format
    img_array = np.array(img)
    current_analysis = analyze_image_data(img_array)
    
    return img, current_analysis

def apply_ikeda_processing(img, mode="blackwhite", threshold=0.5):
    """
    Apply Ikeda-style processing to image
    """
    img_array = np.array(img.convert("RGB"))
    
    if mode == "blackwhite":
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Apply threshold
        _, binary = cv2.threshold(gray, int(threshold * 255), 255, cv2.THRESH_BINARY)
        # Convert back to RGB
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    elif mode == "grid":
        # Quantize to grid
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        grid_size = 16
        h, w = gray.shape
        quantized = np.zeros_like(gray)
        
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                region = gray[y:y+grid_size, x:x+grid_size]
                avg_value = np.mean(region)
                quantized[y:y+grid_size, x:x+grid_size] = avg_value
        
        # Apply threshold and convert
        _, binary = cv2.threshold(quantized, int(threshold * 255), 255, cv2.THRESH_BINARY)
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    elif mode == "data":
        # Create data overlay pattern
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Create pattern
        pattern = np.zeros_like(gray)
        
        # Add edges
        pattern = np.maximum(pattern, edges)
        
        # Add grid overlay
        h, w = pattern.shape
        for y in range(0, h, 32):
            pattern[y, :] = 255
        for x in range(0, w, 32):
            pattern[:, x] = 255
        
        result = cv2.cvtColor(pattern, cv2.COLOR_GRAY2RGB)
    
    else:
        result = img_array
    
    return Image.fromarray(result)

# ==================== WEBSOCKET SERVER ====================

WS_HOST = "localhost"
WS_PORT = 5010
SEND_DELAY = 0.1  # Fast updates for Ikeda aesthetic

class IkedaImageServer:
    def __init__(self):
        self.current_mode = "data"  # Default to data visualization mode
        self.threshold = 0.5
        self.analysis_history = []
        
    def generate_ikeda_image(self):
        """Generate image with embedded data analysis"""
        
        # Create data-driven image
        img, analysis = create_data_visualization_image()
        
        # Apply Ikeda processing
        processed_img = apply_ikeda_processing(img, self.current_mode, self.threshold)
        
        # Store analysis for trends
        self.analysis_history.append(analysis)
        if len(self.analysis_history) > 100:  # Keep last 100 analyses
            self.analysis_history.pop(0)
        
        img_buffer = io.BytesIO()
        processed_img.save(img_buffer, format='PNG')
        return Artifact(
            kind="image",
            payload=img_buffer.getvalue(),
            width=processed_img.width,
            height=processed_img.height,
            mime="image/png",
            producer="generated_ikeda",
            score=1.0,
            novelty=1.0,
            rights=RightsMetadata(
                status="known",
                license="generated",
                creator="ImageFlasherWGPU",
                transformation="procedurally generated Ikeda visualization",
            ),
            metadata={
                "analysis": analysis,
                "mode": self.current_mode,
                "threshold": self.threshold,
            },
        )
    
    async def produce(self, broker: ArtifactBroker):
        """Generate independently of viewers and publish to the shared ring."""
        frame_count = 0
        start_time = time.time()
        while True:
            await broker.emit(await asyncio.to_thread(self.generate_ikeda_image))
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                logger.info("Published %s generated artifacts, %.1f FPS avg", frame_count, frame_count / elapsed)
            await asyncio.sleep(SEND_DELAY)

async def main(host: str = WS_HOST, port: int = WS_PORT):
    server = IkedaImageServer()
    broker = ArtifactBroker(host, port, capacity=256, client_queue_size=32)
    
    logger.info("Starting Ikeda Image Server on ws://%s:%s", host, port)
    logger.info("Generating data-driven black & white visualizations")
    
    producer = asyncio.create_task(server.produce(broker))
    async with broker.serve():
        try:
            await asyncio.Future()
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procedural Ikeda artifact source")
    parser.add_argument("--host", default=WS_HOST)
    parser.add_argument("--port", type=int, default=WS_PORT)
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port))
