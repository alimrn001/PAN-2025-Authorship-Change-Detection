import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from gensim.models.fasttext import load_facebook_model
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

SET_TYPE = "hard"
MAX_LEN = 100

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    print("Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    device = torch.device("cpu")
    print("Using CPU")

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

# Dataset classes
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
        return torch.tensor(s1_vec, dtype=torch.float32), torch.tensor(s2_vec, dtype=torch.float32), label, item['s1'], item['s2']

class PairwiseDataset(Dataset):
    def __init__(self, npy_file):
        self.data = np.load(npy_file, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['s1'], item['s2'], item['label']

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
        _, (h_n, _) = self.lstm(x)
        h_forward = h_n[0]
        h_backward = h_n[1]
        return torch.cat((h_forward, h_backward), dim=1)

    def forward(self, s1, s2):
        s1_encoded = self.encode(s1)
        s2_encoded = self.encode(s2)
        diff = torch.abs(s1_encoded - s2_encoded)
        return self.fc(diff)

# CrossAttnLaBSE model
class CrossAttnLaBSE(nn.Module):
    def __init__(self, model_name="sentence-transformers/LaBSE", dropout_rate=0.1, num_heads=8):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden * 5, 2)

    def forward(self, s1_list, s2_list):
        enc1 = self.tokenizer(s1_list, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        enc2 = self.tokenizer(s2_list, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        out1 = self.bert(**enc1).last_hidden_state
        out2 = self.bert(**enc2).last_hidden_state
        cls1 = out1[:, 0, :]
        cls2 = out2[:, 0, :]
        cross12 = self.cross_attn(query=cls1.unsqueeze(1), key=out2, value=out2)[0].squeeze(1)
        cross21 = self.cross_attn(query=cls2.unsqueeze(1), key=out1, value=out1)[0].squeeze(1)
        diff = torch.abs(cls1 - cls2)
        fused = torch.cat([cls1, cls2, diff, cross12, cross21], dim=1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits

# Function to get model predictions
def get_model_predictions(model, loader, model_type="siamese"):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Generating {model_type} predictions", leave=False):
            if model_type == "siamese":
                s1_vec, s2_vec, labels, _, _ = batch
                s1_vec, s2_vec = s1_vec.to(device), s2_vec.to(device)
                logits = model(s1_vec, s2_vec)
            else:
                _, _, labels, s1_text, s2_text = batch
                logits = model(s1_text, s2_text)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
    return np.vstack(all_probs), np.array(all_labels)

# Confusion matrix plotting
def plot_confusion_matrix(true_labels, pred_labels, name):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"ensemble_CM/{SET_TYPE}_test_{name}_confusion_matrix.png")
    plt.close()

# Main function
def main():
    print(f"Training ensemble on {SET_TYPE} set...")

    # Load datasets
    dataset = PairwiseFastTextDataset(f"../../dataset/initial/embeddings/main/{SET_TYPE}_train.npy", max_len=MAX_LEN)
    valid_dataset = PairwiseFastTextDataset(f"../../dataset/initial/embeddings/main/{SET_TYPE}_validation.npy", max_len=MAX_LEN)
    test_dataset = PairwiseFastTextDataset(f"../../dataset/initial/embeddings/main/{SET_TYPE}_test.npy", max_len=MAX_LEN)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load pre-trained models
    print("Loading SiameseLSTM model...")
    siamese_model = SiameseLSTM().to(device)
    siamese_model.load_state_dict(torch.load(f"../params/siamese_v2_{SET_TYPE}.pth", map_location=device))

    print("Loading CrossAttnLaBSE model...")
    labse_model = CrossAttnLaBSE().to(device)
    labse_model.load_state_dict(torch.load(f"../params/labse_finetune_v4_model_{SET_TYPE}.pth", map_location=device))

    # Generate predictions
    print("Generating predictions for SiameseLSTM...")
    siamese_train_probs, train_labels = get_model_predictions(siamese_model, train_loader, model_type="siamese")
    siamese_valid_probs, valid_labels = get_model_predictions(siamese_model, valid_loader, model_type="siamese")
    siamese_test_probs, test_labels = get_model_predictions(siamese_model, test_loader, model_type="siamese")

    print("Generating predictions for CrossAttnLaBSE...")
    labse_train_probs, _ = get_model_predictions(labse_model, train_loader, model_type="labse")
    labse_valid_probs, _ = get_model_predictions(labse_model, valid_loader, model_type="labse")
    labse_test_probs, _ = get_model_predictions(labse_model, test_loader, model_type="labse")

    # Combine features
    train_features = np.hstack([siamese_train_probs, labse_train_probs])
    valid_features = np.hstack([siamese_valid_probs, labse_valid_probs])
    test_features = np.hstack([siamese_test_probs, labse_test_probs])

    # Train XGBoost
    print("Training XGBoost classifier...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric="logloss", random_state=42)
    xgb_model.fit(train_features, train_labels, eval_set=[(valid_features, valid_labels)], verbose=False)
    joblib.dump(xgb_model, f"ensemble_{SET_TYPE}.pkl")
    
    # Evaluation function
    def evaluate_set(features, labels, set_name):
        preds = xgb_model.predict(features)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        print(f"{set_name} Set Results:")
        print(f"Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")
        return preds

    # Evaluate on all sets
    print("\n=== Ensemble Results ===")
    train_preds = evaluate_set(train_features, train_labels, "Train")
    valid_preds = evaluate_set(valid_features, valid_labels, "Validation")
    test_preds = evaluate_set(test_features, test_labels, "Test")

    # Classification report for test set
    print("\nTest Set Classification Report:")
    print(classification_report(test_labels, test_preds, digits=4))

    # Plot confusion matrix for test set
    plot_confusion_matrix(test_labels, test_preds, "ensemble")

if __name__ == "__main__":
    main()