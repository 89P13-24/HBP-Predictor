"""
Script 2: Extract ESM-2 Embeddings
==================================
Extracts protein embeddings using pre-trained ESM-2 model.

Installation:
    pip install fair-esm torch tqdm biopython

Usage:
    # Default (650M model, batch_size=4)
    python extract_embeddings.py

    # Faster smaller model
    python extract_embeddings.py --model esm2_t12_35M_UR50D --batch_size 16

    # Highest quality (requires 16GB+ GPU)
    python extract_embeddings.py --model esm2_t36_3B_UR50D --batch_size 2

Input:
    - heme_dataset_uniprot.csv (from Script 1)

Output:
    - protein_embeddings.pt: PyTorch tensor file containing embeddings and labels
"""

import torch
import argparse
import pandas as pd
from tqdm import tqdm

class ESM2EmbeddingExtractor:
    """Extract embeddings using ESM-2 pretrained models."""

    def __init__(self, model_name="esm2_t33_650M_UR50D"):
        """
        Available models:
            esm2_t6_8M_UR50D  - Fastest (8M params)
            esm2_t12_35M_UR50D - Fast (35M params)
            esm2_t33_650M_UR50D - Recommended (650M params)
            esm2_t36_3B_UR50D - Best quality (3B params, 16GB+ GPU)
        """
        print(f"Loading ESM-2 model: {model_name}...")

        try:
            import esm
        except ImportError:
            print("❌ ERROR: fair-esm not installed.")
            print("Install via: pip install fair-esm")
            raise

        # Load chosen model
        model_map = {
            "esm2_t6_8M_UR50D": (esm.pretrained.esm2_t6_8M_UR50D, 6),
            "esm2_t12_35M_UR50D": (esm.pretrained.esm2_t12_35M_UR50D, 12),
            "esm2_t30_150M_UR50D": (esm.pretrained.esm2_t30_150M_UR50D, 30),
            "esm2_t33_650M_UR50D": (esm.pretrained.esm2_t33_650M_UR50D, 33),
            "esm2_t36_3B_UR50D": (esm.pretrained.esm2_t36_3B_UR50D, 36)
        }

        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")

        loader, self.repr_layer = model_map[model_name]
        self.model, self.alphabet = loader()
        self.batch_converter = self.alphabet.get_batch_converter()

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✓ Model loaded on {self.device}")
        print(f"  Embedding dimension: {self.model.embed_dim}")

    def extract_embeddings(self, df: pd.DataFrame, batch_size=8):
        """
        Extract per-protein embeddings using mean pooling.
        """
        sequences = df.to_dict("records")
        embeddings_list, labels_list, ids_list = [], [], []

        print(f"\nExtracting embeddings for {len(sequences)} proteins...")

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size)):
                batch = sequences[i:i + batch_size]
                batch_data = [(seq["protein_id"], seq["sequence"]) for seq in batch]
                batch_labels = [seq["label"] for seq in batch]
                batch_ids = [seq["protein_id"] for seq in batch]

                try:
                    _, _, batch_tokens = self.batch_converter(batch_data)
                    batch_tokens = batch_tokens.to(self.device)

                    results = self.model(batch_tokens, repr_layers=[self.repr_layer])
                    token_reprs = results["representations"][self.repr_layer]

                    for j, seq in enumerate(batch):
                        seq_len = len(seq["sequence"])
                        seq_emb = token_reprs[j, 1:seq_len + 1].mean(0)
                        embeddings_list.append(seq_emb.cpu())

                    labels_list.extend(batch_labels)
                    ids_list.extend(batch_ids)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n OOM error! Try: --batch_size {batch_size // 2}")
                        torch.cuda.empty_cache()
                        continue
                    raise e

        embeddings = torch.stack(embeddings_list)
        labels = torch.tensor(labels_list)

        print(f"\n✓ Embeddings extracted: {embeddings.shape}")
        return embeddings, labels, ids_list


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-2 embeddings")
    parser.add_argument("--input", type=str, default="heme_dataset_uniprot.csv",
                        help="Input CSV file from UniProt fetch script")
    parser.add_argument("--output", type=str, default="protein_embeddings.pt",
                        help="Output PyTorch tensor file")
    parser.add_argument("--model", type=str, default="esm2_t33_650M_UR50D",
                        choices=[
                            "esm2_t6_8M_UR50D",
                            "esm2_t12_35M_UR50D",
                            "esm2_t30_150M_UR50D",
                            "esm2_t33_650M_UR50D",
                            "esm2_t36_3B_UR50D"
                        ])
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    print("=" * 70)
    print("ESM-2 EMBEDDING EXTRACTION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)

    df = pd.read_csv(args.input)
    print(f"✓ Loaded {len(df)} sequences from {args.input}")

    extractor = ESM2EmbeddingExtractor(model_name=args.model)
    embeddings, labels, ids = extractor.extract_embeddings(df, args.batch_size)

    data = {
        "embeddings": embeddings,
        "labels": labels,
        "ids": ids,
        "embedding_dim": embeddings.shape[1],
        "num_samples": embeddings.shape[0]
    }

    torch.save(data, args.output)
    print(f"\n✅ Saved embeddings to {args.output}")
    print("\nStatistics:")
    print(f"  Total: {len(labels)}")
    print(f"  Heme-binding: {(labels == 1).sum().item()}")
    print(f"  Non-heme-binding: {(labels == 0).sum().item()}")
    print("\nNext step: python train_model.py")


if __name__ == "__main__":
    main()
