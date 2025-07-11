# This version uses a BiLSTM feeding token-level embeddings to it generated by FastText

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from gensim.models.fasttext import load_facebook_model
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

if torch.cuda.is_available():
    device = 1
    torch.cuda.set_device(device) 

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Using CPU")


MAX_LEN = 100
SET_TYPE = "medium"

# Load FastText model
print("Loading FastText model...")
fasttext_model = load_facebook_model("../../dataset/FastText/crawl-300d-2M-subword.bin").wv
embedding_dim = fasttext_model.vector_size

# Tokenizer function
def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens

# Sentence to vector
def sentence_to_vector(sentence, max_len):
    tokens = tokenize(sentence)
    vecs = []
    for token in tokens[:max_len]:
        try:
            vec = fasttext_model[token]
        except KeyError:
            vec = np.zeros(embedding_dim)
        vecs.append(vec)
    while len(vecs) < max_len:
        vecs.append(np.zeros(embedding_dim))
    return np.array(vecs)

# Dataset class
class PairwiseFastTextDataset(Dataset):
    def __init__(self, npy_path, max_len=50):
        self.data = np.load(npy_path, allow_pickle=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        s1_vec = sentence_to_vector(item['s1'], self.max_len)
        s2_vec = sentence_to_vector(item['s2'], self.max_len)
        label = int(item['label'])
        return torch.tensor(s1_vec, dtype=torch.float32), torch.tensor(s2_vec, dtype=torch.float32), label

# Siamese LSTM model
class SiameseLSTM(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1):
        super(SiameseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def encode(self, x):
        _, (h_n, _) = self.lstm(x)  # h_n shape: (2, B, H)
        h_forward = h_n[0]
        h_backward = h_n[1]
        return torch.cat((h_forward, h_backward), dim=1)

    def forward(self, s1, s2):
        s1_encoded = self.encode(s1)
        s2_encoded = self.encode(s2)
        diff = torch.abs(s1_encoded - s2_encoded)
        return self.fc(diff)

# Metrics
def compute_metrics(logits, labels):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1, preds

# Training function
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for s1, s2, labels in tqdm(dataloader, desc="Training", leave=False):
        s1, s2 = s1.to(device), s2.to(device)
        labels = torch.tensor(labels).to(device)
        optimizer.zero_grad()
        logits = model(s1, s2)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc, f1, preds = compute_metrics(logits, labels)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro")

# Evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for s1, s2, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            s1, s2 = s1.to(device), s2.to(device)
            labels = torch.tensor(labels).to(device)
            logits = model(s1, s2)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            acc, f1, preds = compute_metrics(logits, labels)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro"), all_preds, all_labels

# Confusion matrix
def plot_confusion_matrix(true_labels, pred_labels, name):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"siamese_v2_CM/{SET_TYPE}_test_{name}_confusion_matrix.png")
    plt.close()

# Main
def main():
    print(f"Training on {SET_TYPE} set...")

    train_dataset = PairwiseFastTextDataset(f"../../dataset/initial/embeddings/main/{SET_TYPE}_train.npy", max_len=MAX_LEN)
    valid_dataset = PairwiseFastTextDataset(f"../../dataset/initial/embeddings/main/{SET_TYPE}_validation.npy", max_len=MAX_LEN)
    test_dataset = PairwiseFastTextDataset(f"../../dataset/initial/embeddings/main/{SET_TYPE}_test.npy", max_len=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = SiameseLSTM().to(device)
    class_weights = torch.tensor([1.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training FastText-based Siamese LSTM model...\n")
    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")
        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, valid_loader, criterion)
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

    torch.save(model.state_dict(), f"../params/siamese_v2_{SET_TYPE}.pth")

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion)
    print("\nTest Set Results:")
    print(f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, digits=4))

    plot_confusion_matrix(test_labels, test_preds, "fasttext")

if __name__ == "__main__":
    main()
