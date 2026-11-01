from diffusers.utils import load_image
from PIL import Image

def get_width_height(w, h):
    w = (w // 8) * 8
    h = (h // 8) * 8
    return w, h

def resize_img(img_path, upscale_times):
    img = load_image(img_path)
    if upscale_times <=0:
        return img
    w, h = img.size
    w = w * upscale_times
    h = h * upscale_times
    w, h = get_width_height(int(w), int(h))
    img = img.resize(
        (w, h),
        resample = Image.LANCZOS if upscale_times > 1 else Image.AREA
    )
    return img
