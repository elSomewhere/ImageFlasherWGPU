import asyncio
import os
import io
import random
from typing import Optional

import websockets
from PIL import Image, ImageDraw, ImageChops, ImageOps
import numpy as np
import cv2

# ------------------------------------------------------------------------------------
# 1) A more sophisticated VHS effect using OpenCV. We wrap it inside a Pillow-based function.
# ------------------------------------------------------------------------------------
def apply_better_vhs_effect(
        frame: np.ndarray,
        luma_bandwidth=5,
        chroma_bandwidth=8,
        luma_noise_level=10.0,
        chroma_noise_level=20.0,
        ghost_shift_pixels=3,
        ghost_strength=0.3,
        line_glitch_probability=0.005,
        line_glitch_strength=0.8
):
    """
    Applies a 'better' VHS effect by simulating bandwidth limitation, color bleed,
    ghosting, noise, and random glitch lines on an image (in BGR NumPy array form).

    Parameters
    ----------
    frame : np.ndarray
        Input image in BGR format.
    luma_bandwidth : int
        Horizontal blur 'window' for simulating luma bandwidth limitation.
    chroma_bandwidth : int
        Horizontal blur 'window' for simulating chroma bandwidth limitation.
    luma_noise_level : float
        Standard deviation of random noise on the luminance channel.
    chroma_noise_level : float
        Standard deviation of random noise on the chrominance channels.
    ghost_shift_pixels : int
        Number of horizontal pixels to shift the ghost/edge image.
    ghost_strength : float
        How strongly to overlay the ghost (0.0 to 1.0).
    line_glitch_probability : float
        Probability that a given row will contain a “tracking glitch” or distortion.
    line_glitch_strength : float
        Magnitude of the glitch distortion shift in a row.

    Returns
    -------
    np.ndarray
        Output BGR image with simulated VHS artifacts.
    """

    # Convert to YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y, Cr, Cb = cv2.split(ycrcb)

    # -------------------------------------------------------------------
    # LUMA PROCESSING: simulate limited bandwidth by horizontal blur
    # -------------------------------------------------------------------
    if luma_bandwidth > 1:
        kernel_luma = np.ones(luma_bandwidth, dtype=np.float32) / luma_bandwidth
        for row_idx in range(Y.shape[0]):
            Y[row_idx, :] = np.convolve(Y[row_idx, :], kernel_luma, mode='same')

    # -------------------------------------------------------------------
    # CHROMA PROCESSING: color bleed, reduce bandwidth, offset
    # -------------------------------------------------------------------
    if chroma_bandwidth > 1:
        kernel_chroma = np.ones(chroma_bandwidth, dtype=np.float32) / chroma_bandwidth
        for row_idx in range(Cr.shape[0]):
            Cr[row_idx, :] = np.convolve(Cr[row_idx, :], kernel_chroma, mode='same')
            Cb[row_idx, :] = np.convolve(Cb[row_idx, :], kernel_chroma, mode='same')

    def shift_channel(channel, shift):
        """Shift channel horizontally by 'shift' pixels."""
        channel_shifted = np.zeros_like(channel)
        if shift >= 0:
            channel_shifted[:, shift:] = channel[:, :-shift]
        else:
            channel_shifted[:, :shift] = channel[:, -shift:]
        return channel_shifted

    # Mild color offset: shift Cr and Cb in opposite directions
    Cr = shift_channel(Cr, shift=2)
    Cb = shift_channel(Cb, shift=-2)

    # -------------------------------------------------------------------
    # SIGNAL REFLECTIONS / ghosting
    # -------------------------------------------------------------------
    grad_x = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3)
    grad_x_abs = np.absolute(grad_x)
    ghost = shift_channel(grad_x_abs, ghost_shift_pixels)
    Y = cv2.addWeighted(Y, 1.0, ghost, ghost_strength, 0)

    # -------------------------------------------------------------------
    # NOISE & GLITCH
    # -------------------------------------------------------------------
    # Random noise
    noise_luma = np.random.normal(0, luma_noise_level, Y.shape).astype(np.float32)
    noise_cr = np.random.normal(0, chroma_noise_level, Cr.shape).astype(np.float32)
    noise_cb = np.random.normal(0, chroma_noise_level, Cb.shape).astype(np.float32)
    Y += noise_luma
    Cr += noise_cr
    Cb += noise_cb

    # Random horizontal glitch lines
    h, w = Y.shape
    for row in range(h):
        if random.random() < line_glitch_probability:
            glitch_offset = int(line_glitch_strength * (random.random() - 0.5) * w)
            Y[row, :] = np.roll(Y[row, :], glitch_offset)
            Cr[row, :] = np.roll(Cr[row, :], glitch_offset)
            Cb[row, :] = np.roll(Cb[row, :], glitch_offset)

    # Clip
    Y = np.clip(Y, 0, 255)
    Cr = np.clip(Cr, 0, 255)
    Cb = np.clip(Cb, 0, 255)

    # Recombine YCrCb -> BGR
    ycrcb_vhs = cv2.merge([Y, Cr, Cb]).astype(np.uint8)
    out_bgr = cv2.cvtColor(ycrcb_vhs, cv2.COLOR_YCrCb2BGR)
    return out_bgr

def apply_vhs_filter_pil(img: Image.Image) -> Image.Image:
    """
    Applies a more advanced 'VHS/broken TV' style effect using OpenCV
    but returns a PIL Image.

    1) Convert PIL image to BGR NumPy array
    2) Apply VHS effect
    3) Convert back to PIL
    """
    # Convert PIL -> NumPy (RGB)
    img_rgb = np.array(img.convert("RGB"))
    # Convert RGB -> BGR for OpenCV
    frame_bgr = img_rgb[:, :, ::-1]

    # Apply the advanced VHS effect
    result_bgr = apply_better_vhs_effect(
        frame_bgr,
        luma_bandwidth=5,
        chroma_bandwidth=8,
        luma_noise_level=10.0,
        chroma_noise_level=20.0,
        ghost_shift_pixels=3,
        ghost_strength=0.3,
        line_glitch_probability=0.005,
        line_glitch_strength=0.8
    )

    # Convert back BGR -> RGB -> PIL
    result_rgb = result_bgr[:, :, ::-1]
    result_pil = Image.fromarray(result_rgb, mode="RGB")
    return result_pil


# ------------------------------------------------------------------------------------
# 2) A function to create random images from scratch each time.
# ------------------------------------------------------------------------------------
def generate_random_image(width=512, height=512, use_vhs_filter=True) -> bytes:
    """
    Creates a random image with a random background color or gradient,
    draws a few random shapes, optionally applies VHS filter,
    then returns the PNG bytes.
    """
    # Create a new image
    img = Image.new("RGB", (width, height), color=(0, 0, 0))

    # Optionally, make a random gradient background
    color_a = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    color_b = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for y in range(height):
        ratio = y / (height - 1)
        r = int(color_a[0] * (1 - ratio) + color_b[0] * ratio)
        g = int(color_a[1] * (1 - ratio) + color_b[1] * ratio)
        b = int(color_a[2] * (1 - ratio) + color_b[2] * ratio)
        for x in range(width):
            img.putpixel((x, y), (r, g, b))

    # Draw random shapes
    draw = ImageDraw.Draw(img)
    for _ in range(5):  # 5 random shapes
        shape_type = random.choice(["rectangle", "ellipse", "triangle"])
        x1 = random.randint(0, width - 1)
        y1 = random.randint(0, height - 1)
        x2 = random.randint(x1, width)
        y2 = random.randint(y1, height)

        fill_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(50, 200),  # alpha
        )

        if shape_type == "rectangle":
            draw.rectangle([x1, y1, x2, y2], fill=fill_color)
        elif shape_type == "ellipse":
            draw.ellipse([x1, y1, x2, y2], fill=fill_color)
        else:  # triangle
            x3 = random.randint(0, width)
            y3 = random.randint(0, height)
            draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=fill_color)

    # Optionally apply VHS filter
    if use_vhs_filter:
        img = apply_vhs_filter_pil(img)

    # Convert to PNG bytes
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


# ------------------------------------------------------------------------------------
# 3) WebSocket server that sends random images
# ------------------------------------------------------------------------------------
WS_HOST = "localhost"
WS_PORT = 5010

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

SEND_DELAY = 0.1  # seconds between sends


async def image_sender(websocket):
    """
    Repeatedly generate random images and send them to the client every SEND_DELAY seconds.
    """
    print("WebSocket client connected, starting random image generation.")
    try:
        while True:
            # Generate random image bytes
            img_data = generate_random_image(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, use_vhs_filter=True)
            await websocket.send(img_data)
            print("Sent a random image.")
            await asyncio.sleep(SEND_DELAY)
    except websockets.ConnectionClosed:
        print("Client disconnected.")
    except Exception as e:
        print("Unhandled error in image_sender:", e)


async def main():
    print(f"Starting WebSocket server on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(image_sender, WS_HOST, WS_PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())