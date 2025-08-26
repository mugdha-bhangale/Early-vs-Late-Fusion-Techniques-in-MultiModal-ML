import re
import pandas as pd
from torchvision import transforms
from PIL import Image

# --- text preprocessing ---
def clean_text(text, max_words=256):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"[^a-z0-9\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words)

# --- image preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_and_process_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return transform(img)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
