import torch.nn as nn
import torch

class EarlyFusionModel(nn.Module):
    def __init__(self, bert, resnet, num_classes=5):
        super().__init__()
        self.bert = bert
        self.resnet = resnet

        d_text, d_img = 768, 2048
        d_fused = d_text + d_img

        self.classifier = nn.Sequential(
            nn.Linear(d_fused, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, images):
        text_feats = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        img_feats = self.resnet(images).view(images.size(0), -1)
        fused = torch.cat([text_feats, img_feats], dim=1)
        return self.classifier(fused)
