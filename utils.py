import os

def get_image_path(asin, images_dir):
    path = os.path.join(images_dir, asin + ".jpg")
    return path if os.path.exists(path) else None
