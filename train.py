import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------
# Dataset + Collate
# -----------------------
class ProteinDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def collate_batch(batch, max_len=None):
    """Pads/truncates variable-length [L, D] tensors and returns mask."""
    embeddings, labels = zip(*batch)
    dim = embeddings[0].shape[1]

    # Clip sequence length if specified
    if max_len is None:
        max_len = max(e.shape[0] for e in embeddings)
    else:
        max_len = min(max_len, max(e.shape[0] for e in embeddings))

    padded = torch.zeros(len(embeddings), max_len, dim)
    mask = torch.zeros(len(embeddings), max_len, dtype=torch.bool)

    for i, e in enumerate(embeddings):
        seq = e[:max_len]  # truncate if longer
        L = seq.shape[0]
        padded[i, :L] = seq
        mask[i, :L] = 1

    return padded, torch.tensor(labels), mask


# -----------------------
# Model
# -----------------------
class AttentionHemeClassifier(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, num_heads=8, dropout=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attn_pool = nn.Linear(input_dim, 1)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim // 2, 2)
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)

    def forward(self, x, mask):
        """
        x: [B, L, D]
        mask: [B, L] (True = valid token)
        """
        attn_out, _ = self.attention(x, x, x, key_padding_mask=~mask)
        x = self.ln1(x + attn_out)

        # ✅ Compute per-residue attention scores
        attn_scores = self.attn_pool(x).squeeze(-1)  # [B, L]
        attn_scores[~mask] = float('-inf')  # mask padding residues
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, L]

        # ✅ Weighted mean pooling
        pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)  # [B, D]

        x = self.ffn(pooled)
        x = self.ln2(x)
        logits = self.classifier(x)

        return logits, attn_weights  # return weights for interpretability


# -----------------------
# Trainer
# -----------------------
class HemeTrainer:
    def __init__(self, model, optimizer, criterion, device, patience=5):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.patience = patience

    def train(self, train_loader, val_loader, epochs):
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for embeddings, labels, mask in train_loader:
                embeddings, labels, mask = embeddings.to(self.device), labels.to(self.device), mask.to(self.device)
                self.optimizer.zero_grad()

                logits, _ = self.model(embeddings, mask)
                loss = self.criterion(logits, labels.long())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            val_loss = self.validate(val_loader)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_attention_model.pt")
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print("Early stopping.")
                break

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for embeddings, labels, mask in val_loader:
                embeddings, labels, mask = embeddings.to(self.device), labels.to(self.device), mask.to(self.device)
                logits, _ = self.model(embeddings, mask)
                loss = self.criterion(logits, labels.long())
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def test(self, test_loader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for embeddings, labels, mask in test_loader:
                embeddings, labels, mask = embeddings.to(self.device), labels.to(self.device), mask.to(self.device)
                logits, _ = self.model(embeddings, mask)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, digits=4))


# -----------------------
# Main
# -----------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data = torch.load(args.data_path)
    embeddings = data["embeddings"]  # list of [L, D] tensors
    labels = np.array(data["labels"])

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        embeddings, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Define collate function with max_len
    collate_fn = lambda batch: collate_batch(batch, max_len=args.max_len)

    train_dataset = ProteinDataset(X_train, y_train)
    val_dataset = ProteinDataset(X_val, y_val)
    test_dataset = ProteinDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    model = AttentionHemeClassifier(input_dim=args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    trainer = HemeTrainer(model, optimizer, criterion, device, patience=args.patience)
    trainer.train(train_loader, val_loader, args.epochs)
    trainer.model.load_state_dict(torch.load("best_attention_model.pt"))
    trainer.test(test_loader)


# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="heme_embeddings.pt")
    parser.add_argument("--embedding_dim", type=int, default=1280)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length for truncation/padding")

    args = parser.parse_args()
    main(args)
