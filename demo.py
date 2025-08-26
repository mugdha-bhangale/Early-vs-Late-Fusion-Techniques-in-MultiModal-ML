import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

from transformers import BertTokenizer, BertModel


# -----------------------------
# 1. text preprocessing
# -----------------------------
def clean_text(text, max_words=256):
    """Lowercase, remove symbols, keep alphanum + basic spaces, trim length"""
    if pd.isnull(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)        # remove HTML tags
    text = re.sub(r"&\w+;", " ", text)        # remove HTML entities
    text = re.sub(r"[^a-z0-9\s.,!?]", " ", text)  # keep letters, numbers, .,!?
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace

    # limit text length (by words)
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words)


# -----------------------------
# 2. load dataset
# -----------------------------
df = pd.read_csv("data/Amazon_reviews_2023.csv")

# keep only rows with image URLs
df = df[df["images_url"].astype(str) != "[]"].reset_index(drop=True)

# drop unused columns
drop_cols = ["title", "user_id", "timestamp", "helpful_vote",
             "verified_purchase", "parent_asin"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# apply text cleaning
src_col = "reviewText" if "reviewText" in df.columns else "review_text"
if src_col in df.columns:
    df["clean_text"] = df[src_col].apply(clean_text)


# -----------------------------
# 3. link images
# -----------------------------
images_dir = os.path.join("data", "images_2")

def get_image_path(asin):
    path = os.path.join(images_dir, asin + ".jpg")
    return path if os.path.exists(path) else None

df["images_path"] = df["asin"].apply(get_image_path)

# drop rows with missing local images
df = df.dropna(subset=["images_path"]).reset_index(drop=True)

print("Dataset size:", len(df))
print("With images:", df["images_path"].notnull().sum())


# -----------------------------
# 4. train-test split
# -----------------------------
train, test = train_test_split(df, test_size=0.2,
                               random_state=42, shuffle=True)
print("Train shape:", train.shape)
print("Test shape:", test.shape)


# -----------------------------
# 5. image preprocessing
# -----------------------------
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


# -----------------------------
# 6. device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# 7. text feature extractor (BERT)
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

def get_text_embedding(text_list, max_len=128):
    inputs = tokenizer(text_list, return_tensors="pt",
                       padding=True, truncation=True,
                       max_length=max_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS]
    return cls_embeddings.cpu()  # [B, 768]


# -----------------------------
# 8. image feature extractor (ResNet50)
# -----------------------------
resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # cut FC
resnet = resnet.to(device)
resnet.eval()

def get_image_embedding(image_paths):
    imgs = []
    for p in image_paths:
        tensor = load_and_process_image(p)
        if tensor is not None:
            imgs.append(tensor)
    if not imgs:
        return None
    imgs = torch.stack(imgs).to(device)
    with torch.no_grad():
        feats = resnet(imgs)                # [B, 2048, 1, 1]
        feats = feats.view(feats.size(0), -1)  # [B, 2048]
    return feats.cpu()


# -----------------------------
# 9. quick test on first 2 samples
# -----------------------------
sample_texts = df["clean_text"].head(2).tolist()
sample_images = df["images_path"].head(2).tolist()

text_feats = get_text_embedding(sample_texts)
image_feats = get_image_embedding(sample_images)

print("Text embeddings shape:", text_feats.shape)    # [2, 768]
print("Image embeddings shape:", image_feats.shape)  # [2, 2048]

torch.save({"text": text_feats, "image": image_feats}, "sample_embeddings.pt")
print("Sample embeddings saved.")

from torch.utils.data import Dataset, DataLoader

class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, transform, max_len=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # text
        text = row["clean_text"]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)       # [L]
        attention_mask = enc["attention_mask"].squeeze(0)  # [L]

        # image
        img_path = row["images_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)                 # [3,224,224]

        # label (ratings are 1–5 → shift to 0–4 classes)
        label = int(row["rating"]) - 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }

# train/test datasets
train_dataset = ReviewDataset(train, tokenizer, transform, max_len=128)
test_dataset = ReviewDataset(test, tokenizer, transform, max_len=128)

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# quick sanity check
batch = next(iter(train_loader))
print(batch["input_ids"].shape)    # [B, L]
print(batch["attention_mask"].shape)
print(batch["image"].shape)        # [B, 3, 224, 224]
print(batch["label"].shape)        # [B]


class EarlyFusionModel(nn.Module):
    def __init__(self, bert, resnet, num_classes=5):
        super().__init__()
        self.bert = bert
        self.resnet = resnet

        # hidden sizes
        d_text, d_img = 768, 2048
        d_fused = d_text + d_img

        self.classifier = nn.Sequential(
            nn.Linear(d_fused, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, images):
        # text → [B,768]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = outputs.last_hidden_state[:, 0, :]

        # image → [B,2048]
        img_feats = self.resnet(images)               # [B,2048,1,1]
        img_feats = img_feats.view(img_feats.size(0), -1)

        # concat
        fused = torch.cat([text_feats, img_feats], dim=1)  # [B,2816]

        return self.classifier(fused)   # [B,num_classes]

# move models to device
fusion_model = EarlyFusionModel(bert_model, resnet, num_classes=5).to(device)

# get a batch
batch = next(iter(train_loader))
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
images = batch["image"].to(device)

# forward pass
with torch.no_grad():
    outputs = fusion_model(input_ids, attention_mask, images)

print("Model output shape:", outputs.shape)  # [B,5]


import torch.nn.functional as F

# get a batch
batch = next(iter(train_loader))
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
images = batch["image"].to(device)
labels = batch["label"]  # ground truth (0–4)

with torch.no_grad():
    outputs = fusion_model(input_ids, attention_mask, images)  # [B,5]

    # convert logits → probabilities
    probs = F.softmax(outputs, dim=1)  # [B,5]

    # predicted class indices (0–4)
    preds = torch.argmax(probs, dim=1)

    # convert to 1–5 ratings
    ratings = preds + 1
print("\nSample predictions:")
for i in range(len(ratings)):
    print(f"Review {i+1}: True Rating = {labels[i].item()+1}, Predicted = {ratings[i].item()}")
