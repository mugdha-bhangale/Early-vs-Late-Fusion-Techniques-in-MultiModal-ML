import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torchvision import models
from torchvision.models import ResNet50_Weights

# --- device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- BERT ---
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

def get_text_embedding(text_list, max_len=128):
    inputs = tokenizer(text_list, return_tensors="pt",
                       padding=True, truncation=True, max_length=max_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B,768]
    return cls_embeddings.cpu()

# --- ResNet50 ---
resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet = nn.Sequential(*list(resnet.children())[:-1]).to(device)
resnet.eval()

def get_image_embedding(images, transform):
    imgs = [transform(img) for img in images if img is not None]
    if not imgs:
        return None
    imgs = torch.stack(imgs).to(device)
    with torch.no_grad():
        feats = resnet(imgs)        # [B,2048,1,1]
        feats = feats.view(feats.size(0), -1)
    return feats.cpu()
