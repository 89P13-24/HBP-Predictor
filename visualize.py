import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse

# ----------------------------
# Define same architecture
# ----------------------------
class AttentionModel(nn.Module):
    def __init__(self, embed_dim=1280, hidden_dim=256, max_len=512):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x, attn_mask=None, return_attn=False):
        # attn_out: [B, L, D], attn_weights: [B, num_heads, L, L]
        attn_out, attn_weights = self.attention(x, x, x, attn_mask=attn_mask, need_weights=True, average_attn_weights=False)
        
        # Mean pooling across sequence
        pooled = attn_out.mean(dim=1)
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        out = torch.sigmoid(self.fc2(x))

        if return_attn:
            return out, attn_weights
        return out


# ----------------------------
# Visualization
# ----------------------------
def visualize_attention(model, embeddings, ids, seq_index=0, max_len=512, save_fig=True):
    model.eval()
    with torch.no_grad():
        seq_emb = embeddings[seq_index]  # [L, D]
        seq_id = ids[seq_index]
        L = min(seq_emb.size(0), max_len)

        # Prepare padded input
        padded = torch.zeros((1, max_len, seq_emb.size(1)))
        padded[0, :L, :] = seq_emb[:L]

        # Run model and get attention weights
        output, attn_weights = model(padded, return_attn=True)
        # attn_weights shape: [1, num_heads, L, L]
        attn_weights = attn_weights[0, :, :L, :L].cpu().numpy()  # remove batch dim

        # Average over heads for visualization
        mean_attn = attn_weights.mean(axis=0)

        # Compute per-residue importance (mean attention received)
        residue_importance = mean_attn.mean(axis=0)  # average over query dimension

    # ----------------------------
    # Plot full attention heatmap
    # ----------------------------
    plt.figure(figsize=(8, 6))
    plt.imshow(mean_attn, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention Weight")
    plt.title(f"Self-Attention Map — Protein ID: {seq_id}")
    plt.xlabel("Key Residue Index")
    plt.ylabel("Query Residue Index")
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"attention_heatmap_{seq_id}.png", dpi=300)
    plt.show()

    # ----------------------------
    # Plot residue-level importance
    # ----------------------------
    plt.figure(figsize=(10, 3))
    plt.plot(residue_importance, color="darkorange", lw=2)
    plt.title(f"Average Attention per Residue — {seq_id}")
    plt.xlabel("Residue Index")
    plt.ylabel("Mean Attention Weight")
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"residue_importance_{seq_id}.png", dpi=300)
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize attention over amino acids")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to saved embeddings .pt file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained attention model .pt file")
    parser.add_argument("--seq_index", type=int, default=0, help="Index of protein to visualize")
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length used during training")
    args = parser.parse_args()

    # Load data
    data = torch.load(args.embeddings)
    embeddings = data["embeddings"]
    ids = data["ids"]

    # Load model weights
    model = AttentionModel(max_len=args.max_len)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))

    # Visualize
    visualize_attention(model, embeddings, ids, seq_index=args.seq_index, max_len=args.max_len)


if __name__ == "__main__":
    main()
