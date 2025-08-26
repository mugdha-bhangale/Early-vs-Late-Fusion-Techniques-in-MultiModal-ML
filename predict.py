import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
from random import sample

from preprocessing import clean_text, transform
from utils import get_image_path
from feature_extractors import tokenizer, bert_model, resnet, device
from dataset import ReviewDataset
from models import EarlyFusionModel


# -----------------------------
# 1. load dataset
# -----------------------------
df = pd.read_csv("data/Amazon_reviews_2023.csv")
df = df[df["images_url"].astype(str) != "[]"].reset_index(drop=True)

# clean text
if "reviewText" in df.columns:
    df["clean_text"] = df["reviewText"].apply(clean_text)

# link images
df["images_path"] = df["asin"].apply(lambda x: get_image_path(x, "data/images_2"))
df = df.dropna(subset=["images_path"]).reset_index(drop=True)

print("Dataset size:", len(df))

# split
_, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# dataset + loader
test_dataset = ReviewDataset(test_df, tokenizer, transform, max_len=128, label_col="rating")
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# -----------------------------
# 2. load trained model
# -----------------------------
model = EarlyFusionModel(bert_model, resnet, num_classes=5).to(device)
model.load_state_dict(torch.load("fusion_model.pth", map_location=device))
model.eval()

print("Loaded trained model ✅")


# -----------------------------
# 3. prediction function
# -----------------------------
def predict_batch(model, batch):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    images = batch["image"].to(device)
    labels = batch["label"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, images)  # [B,5]
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    # convert preds + labels to 1–5 scale
    ratings_pred = preds + 1
    ratings_true = labels + 1
    return ratings_true, ratings_pred


# -----------------------------
# 4. run predictions on random samples
# -----------------------------
# pick random 5 samples from test set
indices = sample(range(len(test_dataset)), 5)
print("\nSample predictions:\n")

for idx in indices:
    item = test_dataset[idx]
    batch = {
        "input_ids": item["input_ids"].unsqueeze(0),
        "attention_mask": item["attention_mask"].unsqueeze(0),
        "image": item["image"].unsqueeze(0),
        "label": item["label"].unsqueeze(0)
    }
    true, pred = predict_batch(model, batch)
    review_text = test_df.iloc[idx]["clean_text"][:100] + "..."  # shorten preview

    print(f"Review: {review_text}")
    print(f"True Rating: {true.item()} | Predicted Rating: {pred.item()}")
    print("-" * 50)
