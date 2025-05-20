# this version uses a lightweight cross-attention layer between sentence embeddings instead of just using [CLS] embeddings and |e1 - e2|
# it uses a transformer encoder block on top of the combined sentence pair.
# it replaces v3 plain “concat+diff” head with a cross‑attention fusion head on top of LaBSE
# five pieces are concatenated: [ cls1, cls2, |cls1–cls2|, cross12, cross21 ]

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, accuracy_score
import sys
from collections import Counter

SET_TYPE = "medium"

if torch.cuda.is_available():
    device = 1
    torch.cuda.set_device(device) 

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Using CPU")

class PairwiseDataset(Dataset):
    def __init__(self, npy_file):
        self.data = np.load(npy_file, allow_pickle=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['s1'], item['s2'], item['label']

class CrossAttnLaBSE(nn.Module):
    def __init__(self,
                 model_name="sentence-transformers/LaBSE",
                 dropout_rate=0.1,
                 num_heads=8):
        super().__init__()
        self.bert      = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        hidden = self.bert.config.hidden_size  
        # Cross‑attention: CLS → other sentence’s tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        # Classifier on [ cls1 | cls2 | |cls1–cls2| | cross12 | cross21 ]
        self.classifier = nn.Linear(hidden * 5, 2)

    def forward(self, s1_list, s2_list):
        enc1 = self.tokenizer(
            s1_list, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(device)
        enc2 = self.tokenizer(
            s2_list, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(device)

        out1 = self.bert(**enc1).last_hidden_state  
        out2 = self.bert(**enc2).last_hidden_state 

        cls1 = out1[:, 0, :] 
        cls2 = out2[:, 0, :] 

        cross12 = self.cross_attn(
            query=cls1.unsqueeze(1),  # (B, 1, H)
            key=out2, value=out2
        )[0].squeeze(1)             # → (B, H)

        cross21 = self.cross_attn(
            query=cls2.unsqueeze(1),  # (B, 1, H)
            key=out1, value=out1
        )[0].squeeze(1)             # → (B, H)

        diff = torch.abs(cls1 - cls2)  # (B, H)
        fused = torch.cat([cls1, cls2, diff, cross12, cross21], dim=1)  # (B, 5H)
        fused = self.dropout(fused)

        logits = self.classifier(fused)  # (B, 2)
        return logits

def compute_metrics(logits, labels):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro")
    return acc, f1, preds

def train(model, loader, optimizer, criterion):
    model.train()
    tot_loss, all_preds, all_labels = 0.0, [], []
    for s1, s2, labs in tqdm(loader, desc="Train", leave=False, file=sys.stdout):
        optimizer.zero_grad()
        labels = torch.tensor(labs).to(device)
        logits = model(s1, s2)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        acc, f1, preds = compute_metrics(logits, labels)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    avg_loss = tot_loss / len(loader)
    return avg_loss, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro")

def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for s1, s2, labs in tqdm(loader, desc="Eval", leave=False, file=sys.stdout):
            labels = torch.tensor(labs).to(device)
            logits = model(s1, s2)
            tot_loss += criterion(logits, labels).item()
            acc, f1, preds = compute_metrics(logits, labels)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    avg_loss = tot_loss / len(loader)
    return avg_loss, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro"), all_labels, all_preds

def main():
    print(f'Training on: {SET_TYPE}')
    train_np = f"../../dataset/initial/embeddings/main/{SET_TYPE}_train.npy"
    val_np   = f"../../dataset/initial/embeddings/main/{SET_TYPE}_validation.npy"
    test_np  = f"../../dataset/initial/embeddings/main/{SET_TYPE}_test.npy"

    train_ds = PairwiseDataset(train_np)
    val_ds   = PairwiseDataset(val_np)
    test_ds  = PairwiseDataset(test_np)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

    counts = Counter([item[2] for item in train_ds])
    total  = sum(counts.values())
    weights = torch.tensor([total / counts[i] for i in range(2)], dtype=torch.float).to(device)
    print("Class weights:", weights.tolist())

    model     = CrossAttnLaBSE().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    epochs = 3
    for ep in range(epochs):
        print(f"\n=== Epoch {ep+1}/{epochs} ===")
        tr_loss, tr_acc, tr_f1 = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion)
        print(f"Train   ► Loss: {tr_loss:.4f} | Acc: {tr_acc:.4f} | F1: {tr_f1:.4f}")
        print(f"Validate► Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

    torch.save(model.state_dict(), f"../labse_finetune_v4_model_{SET_TYPE}.pth")

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion)
    print(f"\nTest ► Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    main()
