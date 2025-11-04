# utils/image_enhancer.py
from PIL import Image, ImageEnhance

def enhance_image(image_path):
    """
    Applies simple post-processing filters:
    brightness, contrast, and sharpness adjustment.
    """
    img = Image.open(image_path)

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)

    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)

    img.save(image_path)
    return image_path
