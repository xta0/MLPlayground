import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from io import BytesIO

# Configuration
URLS_FILE = 'urls.txt'   # Text file containing the URLs
OUTPUT_DIR = 'images' # Folder to save the images
MAX_DOWNLOADS = 100       # Download at most 100 images
MAX_WORKERS = 10          # Number of parallel threads

def download_and_convert_image(idx_url):
    idx, url = idx_url
    filename = os.path.join(OUTPUT_DIR, f'{idx:03}.png')

    if os.path.exists(filename):
        print(f'{filename} already exists. Skipping download.')
        return

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Open image and convert to PNG
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image.save(filename, format='PNG')
        
        print(f'Downloaded and converted {filename}')
    except Exception as e:
        print(f'Failed to download {url}: {e}')

def download_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(URLS_FILE, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    urls = urls[:MAX_DOWNLOADS]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_and_convert_image, (idx, url)) for idx, url in enumerate(urls, start=1)]
        for future in as_completed(futures):
            pass  # Just ensuring all downloads complete

if __name__ == '__main__':
    download_images()
