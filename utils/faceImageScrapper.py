import requests
from PIL import Image
from io import BytesIO
import os

# Create output folder
os.makedirs("imgData", exist_ok=True)

def download_and_process_image(idx):
    url = "https://thispersondoesnotexist.com"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))
        img = img.convert('L')  # convert to grayscale
        img = img.resize((256, 256))  # resize to 256x256

        img.save(f"imgData/face_{idx}.png")
        print(f"[+] Saved face_{idx}.png")
    except Exception as e:
        print(f"[!] Failed to fetch image {idx}: {e}")

# Download 25 images
for i in range(25):
    download_and_process_image(i)
