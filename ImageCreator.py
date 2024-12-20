import asyncio
import websockets
import random
import io
from PIL import Image, ImageDraw

async def image_sender(websocket):
    """Send continuously generated images to the connected client."""
    width, height = 512, 512
    print("start image sender")
    try:
        while True:
            print("start")
            # Create a gradient image from blue to green horizontally
            img = Image.new('RGBA', (width, height), (0, 0, 0, 255))
            pixels = img.load()
            for y in range(height):
                for x in range(width):
                    t = x / (width - 1)
                    r = 0
                    g = int(255 * t)
                    b = int(255 * (1.0 - t))
                    a = 255
                    pixels[x, y] = (r, g, b, a)

            # Draw a few random circles on the image
            draw = ImageDraw.Draw(img)
            num_circles = 5
            for _ in range(num_circles):
                cx = random.randint(0, width - 1)
                cy = random.randint(0, height - 1)
                radius = random.randint(10, 60)
                cr = random.randint(0, 255)
                cg = random.randint(0, 255)
                cb = random.randint(0, 255)
                ca = 255
                # Draw a filled circle
                # PIL doesn't have a direct circle fill by radius and center,
                # but we can use ellipse with bounding box
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(cr, cg, cb, ca))

            # Encode the image as PNG in memory
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_data = buf.getvalue()

            # Send the PNG binary data over WebSocket
            print("sending")
            await websocket.send(img_data)
            print("sending done")

            # Wait a bit before sending the next image
            await asyncio.sleep(0.01)
            print("image sent")
    except websockets.ConnectionClosed:
        print("Client disconnected")
    except Exception as e:
        print("Error:", e)

async def main():
    async with websockets.serve(image_sender, "localhost", 5000):
        print("WebSocket server started at ws://localhost:5000")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
