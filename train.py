import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataset import ReviewDataset
from preprocessing import transform, clean_text
from utils import get_image_path
from feature_extractors import tokenizer, bert_model, resnet, device
from models import EarlyFusionModel

import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import clean_text

# -----------------------------
# 1. load + preprocess dataset
# -----------------------------
df = pd.read_csv("data/Amazon_reviews_2023.csv")
df = df[df["images_url"].astype(str) != "[]"].reset_index(drop=True)



# --- after df = pd.read_csv(...) ---
if "reviewText" in df.columns:
    df["clean_text"] = df["reviewText"].apply(clean_text)
elif "review_text" in df.columns:
    df["clean_text"] = df["review_text"].apply(clean_text)
else:
    raise KeyError("No review text column found in dataset")


# clean text
if "reviewText" in df.columns:
    df["clean_text"] = df["reviewText"].apply(clean_text)

# link images
df["images_path"] = df["asin"].apply(lambda x: get_image_path(x, "data/images_2"))
df = df.dropna(subset=["images_path"]).reset_index(drop=True)

print("Dataset size:", len(df))

# split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# datasets
train_dataset = ReviewDataset(train_df, tokenizer, transform, max_len=128, label_col="rating")
test_dataset = ReviewDataset(test_df, tokenizer, transform, max_len=128, label_col="rating")

# dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# -----------------------------
# 2. model, loss, optimizer
# -----------------------------
model = EarlyFusionModel(bert_model, resnet, num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)


# -----------------------------
# 3. training function
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_preds, total_labels = 0, [], []

    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask, images)  # [B,5]
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        total_preds.extend(preds)
        total_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(total_labels, total_preds)
    return total_loss / len(loader), acc


# -----------------------------
# 4. evaluation function
# -----------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_preds, total_labels = 0, [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            total_preds.extend(preds)
            total_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(total_labels, total_preds)
    return total_loss / len(loader), acc


# -----------------------------
# 5. training loop
# -----------------------------
EPOCHS = 3
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

# save model
torch.save(model.state_dict(), "fusion_model.pth")
print("Model saved to fusion_model.pth")
