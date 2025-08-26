import torch
from torch.utils.data import Dataset
from PIL import Image

class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, transform, max_len=128, label_col="rating"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # text
        text = row["clean_text"]
        enc = self.tokenizer(text,
                             padding="max_length",
                             truncation=True,
                             max_length=self.max_len,
                             return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # image
        img_path = row["images_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # label
        label = int(row[self.label_col]) - 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }
