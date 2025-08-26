import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn.functional as F

from preprocessing import clean_text, transform
from utils import get_image_path
from dataset import ReviewDataset
from feature_extractors import tokenizer, bert_model, resnet, device
from models import EarlyFusionModel

# --- load dataset ---
df = pd.read_csv("data/Amazon_reviews_2023.csv")
df = df[df["images_url"].astype(str) != "[]"].reset_index(drop=True)
from preprocessing import clean_text

if "reviewText" in df.columns:
    df["clean_text"] = df["reviewText"].apply(clean_text)
elif "review_text" in df.columns:
    df["clean_text"] = df["review_text"].apply(clean_text)
else:
    raise KeyError("No review text column found in dataset")


if "reviewText" in df.columns:
    df["clean_text"] = df["reviewText"].apply(clean_text)

df["images_path"] = df["asin"].apply(lambda x: get_image_path(x, "data/images_2"))
df = df.dropna(subset=["images_path"]).reset_index(drop=True)

print("Dataset size:", len(df))

# --- split ---
train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# --- datasets & loaders ---
train_dataset = ReviewDataset(train, tokenizer, transform, max_len=128, label_col="rating")
test_dataset = ReviewDataset(test, tokenizer, transform, max_len=128, label_col="rating")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# --- model ---
fusion_model = EarlyFusionModel(bert_model, resnet, num_classes=5).to(device)

# --- sanity check ---
batch = next(iter(train_loader))
with torch.no_grad():
    outputs = fusion_model(batch["input_ids"].to(device),
                           batch["attention_mask"].to(device),
                           batch["image"].to(device))

probs = F.softmax(outputs, dim=1)
preds = torch.argmax(probs, dim=1) + 1
print("Predicted ratings:", preds)
