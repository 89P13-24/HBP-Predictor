"""
Script 3: Train Heme-Binding Classifier
========================================
Trains attention-based classifier for heme-binding prediction.

Installation:
    pip install torch scikit-learn matplotlib

Usage:
    python train_model.py --epochs 50 --lr 0.0001
    
Output:
    - best_heme_model.pt: Best model checkpoint
    - training_history.pkl: Training history
    - training_curves.png: Loss/accuracy plots
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, matthews_corrcoef
import pickle
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================================
# MODEL
# ============================================================================

class AttentionHemeClassifier(nn.Module):
    """Attention-based classifier"""
    
    def __init__(self, input_dim=1280, hidden_dim=512, num_heads=8, dropout=0.3):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
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
    
    def forward(self, x):
        x = x.unsqueeze(1)
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.ln1(x + attn_out)
        x = x.squeeze(1)
        x = self.ffn(x)
        x = self.ln2(x)
        logits = self.classifier(x)
        return logits, attn_weights


class ProteinDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ============================================================================
# TRAINER
# ============================================================================

class HemePredictor:
    def __init__(self, model, device='mps' if torch.backends.mps.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_acc': [], 'val_auc': []
        }
    
    def train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        
        for embeddings, labels in tqdm(loader, desc="Training"):
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            logits, _ = self.model(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def validate(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        all_preds, all_probs, all_labels = [], [], []
        
        with torch.no_grad():
            for embeddings, labels in tqdm(loader, desc="Validating"):
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                
                logits, _ = self.model(embeddings)
                loss = criterion(logits, labels)
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        auc = roc_auc_score(all_labels, all_probs)
        
        return total_loss / len(loader), acc, auc, all_preds, all_probs, all_labels
    
    def fit(self, train_loader, val_loader, epochs=50, lr=1e-4, patience=10):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        best_auc = 0
        patience_counter = 0
        
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc, val_auc, _, _, _ = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            
            scheduler.step(val_auc)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_auc': best_auc,
                    'epoch': epoch
                }, 'best_heme_model.pt')
                print(f"✓ Best model saved (AUC: {best_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        checkpoint = torch.load('best_heme_model.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✓ Training complete. Best AUC: {best_auc:.4f}")
    
    def evaluate(self, test_loader):
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, test_auc, preds, probs, labels = \
            self.validate(test_loader, criterion)
        
        print("\n" + "="*70)
        print("TEST SET EVALUATION")
        print("="*70)
        print(f"Accuracy: {test_acc:.4f}")
        print(f"AUC-ROC:  {test_auc:.4f}")
        print(f"MCC:      {matthews_corrcoef(labels, preds):.4f}")
        
        print("\n" + classification_report(
            labels, preds,
            target_names=['Non-Heme', 'Heme-Binding'],
            digits=4
        ))
        
        cm = confusion_matrix(labels, preds)
        print("Confusion Matrix:")
        print(f"              Predicted")
        print(f"            Non   Heme")
        print(f"Actual Non  {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       Heme {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        return {
            'accuracy': test_acc,
            'auc': test_auc,
            'mcc': matthews_corrcoef(labels, preds),
            'confusion_matrix': cm
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train', linewidth=2)
    ax1.plot(history['val_loss'], label='Val', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Metrics
    ax2.plot(history['val_acc'], label='Accuracy', linewidth=2)
    ax2.plot(history['val_auc'], label='AUC', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved training_curves.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train heme-binding classifier')
    parser.add_argument('--input', type=str, default='protein_embeddings.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--patience', type=int, default=10)
    
    args = parser.parse_args()
    
    print("="*70)
    print("HEME-BINDING CLASSIFIER TRAINING")
    print("="*70)
    
    # Load data
    data = torch.load(args.input)
    embeddings = data['embeddings']
    labels = data['labels']
    
    print(f"Loaded {len(labels)} samples")
    print(f"Embedding dim: {embeddings.shape[1]}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        embeddings, labels, test_size=0.2, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create dataloaders
    train_dataset = ProteinDataset(X_train, y_train)
    val_dataset = ProteinDataset(X_val, y_val)
    test_dataset = ProteinDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Build model
    model = AttentionHemeClassifier(
        input_dim=embeddings.shape[1],
        hidden_dim=args.hidden_dim
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    predictor = HemePredictor(model)
    predictor.fit(train_loader, val_loader, epochs=args.epochs, lr=args.lr, patience=args.patience)
    
    # Evaluate
    results = predictor.evaluate(test_loader)
    
    # Save
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(predictor.history, f)
    print("\n✓ Saved training_history.pkl")
    
    plot_training(predictor.history)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("Outputs:")
    print("  - best_heme_model.pt")
    print("  - training_history.pkl")
    print("  - training_curves.png")
    print("\nNext: python predict.py --sequence YOURSEQUENCE")


if __name__ == "__main__":
    main()