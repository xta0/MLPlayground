import os
import logging
import sys
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener
from typing import Tuple
import time


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic'}

def is_image_file(path: Path):
    return path.suffix.lower() in SUPPORTED_EXTENSIONS

def convert_single_file(heic_path: str, jpg_path: str, output_quality: int) -> Tuple[str, bool, float]:
    """
    Convert a single HEIC file to JPG format with optional resizing.
    
    #### Args:
        - heic_path (str): Path to the HEIC file.
        - jpg_path (str): Path to save the converted JPG file.
        - output_quality (int): Quality of the output JPG image.
        - resize (tuple, optional): Width and height to resize the image to.

    #### Returns:
        - tuple: Path to the HEIC file, conversion status, and processing time.
    """
    start_time = time.time()
    try:
        with Image.open(heic_path) as image:
            
            # Automatically handle and preserve EXIF metadata
            exif_data = image.info.get("exif")
            image.save(jpg_path, "JPEG", quality=output_quality, exif=exif_data, optimize=True)
            
            # Preserve the original access and modification timestamps
            heic_stat = os.stat(heic_path)
            os.utime(jpg_path, (heic_stat.st_atime, heic_stat.st_mtime))
            
            processing_time = time.time() - start_time
            return heic_path, True, processing_time  # Successful conversion
    except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
        logger.error("Error converting '%s': %s", heic_path, e)
        processing_time = time.time() - start_time
        return heic_path, False, processing_time  # Failed conversion

def convert_to_jpg(src_path: Path, dest_dir: Path, log_path: Path) -> bool:
    dest_filename = f"{src_path.stem}.jpg"
    dest_file = dest_dir / dest_filename

    if dest_file.exists():
        logging.info(f"⚠️ Skipping already processed: {dest_file.name}")
        return False

    try:
        ext = src_path.suffix.lower()
        if ext == '.heic':
            logging.info(f"🔄 Converting HEIC: {src_path}")
            _, status, _ = convert_single_file(src_path, dest_file, 95)
            if status:
                logging.info(f"✅ Saved: {dest_file}")
            else:
                logging.info(f"❌ Conversion failed for {src_path}")
                with open(log_path, "a") as log:
                    log.write(f"{src_path} — Conversion failed\n")
                return False
        else:
            logging.info(f"🔄 Converting image: {src_path}")
            image = Image.open(src_path).convert("RGB")
            image.save(dest_file, format="JPEG", quality=95)
            logging.info(f"✅ Saved: {dest_file}")
        return True

    except (UnidentifiedImageError, Exception) as e:
        logging.info(f"❌ Failed to convert {src_path}: {e}")
        with open(log_path, "a") as log:
            log.write(f"{src_path} — {e}\n")
        return False

def process_images(src_dir: str, dst_dir: str):
    source = Path(src_dir).resolve()
    destination = Path(dst_dir).resolve()
    log_path = destination / "failures.log"
    register_heif_opener()
    if not source.is_dir():
        logging.info(f"❌ Source directory {source} does not exist.")
        sys.exit(1)

    destination.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    count = 0
    for root, _, files in os.walk(source):
        for name in files:
            file_path = Path(root) / name
            if is_image_file(file_path):
                if convert_to_jpg(file_path, destination, log_path):
                    count += 1
            else:
                logging.info(f"🧾 Skipping unsupported: {file_path.name}")

    logging.info(f"\n🎉 Done! Converted {count} image(s) to {destination}")
    if log_path.exists():
        logging.info(f"⚠️ Some failures logged in: {log_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.info("Usage: python convert_all_images_to_jpg_flat.py <source_dir> <dest_dir>")
        sys.exit(1)

    process_images(sys.argv[1], sys.argv[2])
