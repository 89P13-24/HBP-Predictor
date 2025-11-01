"""
Script: Predict Heme-Binding for New Proteins (Attention Version)
=================================================================
Makes predictions for new protein sequences using the trained
AttentionHemeClassifier and ESM-2 embeddings.

Usage:
    # Single sequence
    python predict_attention.py --sequence MKALIVLGL...

    # From FASTA file
    python predict_attention.py --fasta new_proteins.fasta

    # Batch prediction
    python predict_attention.py --fasta proteins.fasta --output predictions.csv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from typing import List, Tuple

from train import AttentionHemeClassifier

# --------------------------------------------------
# PREDICTOR
# --------------------------------------------------
class HemePredictor:
    def __init__(self, model_path='best_attention_model.pt', esm_model='esm2_t33_650M_UR50D', max_len=512):
        print("Loading models...")

        try:
            import esm
            if esm_model == "esm2_t33_650M_UR50D":
                self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                self.repr_layer = 33
            elif esm_model == "esm2_t30_150M_UR50D":
                self.esm_model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
                self.repr_layer = 30
            elif esm_model == "esm2_t12_35M_UR50D":
                self.esm_model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
                self.repr_layer = 12
            else:
                raise ValueError(f"Unsupported ESM model: {esm_model}")
            self.batch_converter = self.alphabet.get_batch_converter()
            self.esm_model.eval()
            print(f"✓ Loaded ESM model: {esm_model}")
        except ImportError:
            raise ImportError("Please install fair-esm: pip install fair-esm")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len

        # Load classifier
        embedding_dim = self.esm_model.embed_dim
        self.classifier = AttentionHemeClassifier(input_dim=embedding_dim)
        self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
        self.classifier.eval()
        self.esm_model = self.esm_model.to(self.device)
        self.classifier = self.classifier.to(self.device)

        print(f"✓ Classifier loaded and moved to {self.device}")

    # -----------------------
    # ESM Embedding Extraction
    # -----------------------
    def extract_embedding(self, sequence: str) -> torch.Tensor:
        with torch.no_grad():
            data = [("protein", sequence)]
            _, _, tokens = self.batch_converter(data)
            tokens = tokens.to(self.device)

            results = self.esm_model(tokens, repr_layers=[self.repr_layer])
            token_repr = results["representations"][self.repr_layer][0, 1:len(sequence) + 1, :]  # exclude CLS, EOS

            L = min(token_repr.size(0), self.max_len)
            emb = torch.zeros((1, self.max_len, token_repr.size(1)), device=self.device)
            mask = torch.zeros((1, self.max_len), dtype=torch.bool, device=self.device)

            emb[0, :L, :] = token_repr[:L]
            mask[0, :L] = 1
        return emb, mask

    # -----------------------
    # Prediction (single)
    # -----------------------
    def predict_single(self, sequence: str):
        emb, mask = self.extract_embedding(sequence)
        with torch.no_grad():
            logits , _ = self.classifier(emb, mask)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        label = "Heme-binding" if pred_class == 1 else "Non-heme-binding"
        return label, confidence

    # -----------------------
    # Batch prediction
    # -----------------------
    def predict_batch(self, sequences: List[Tuple[str, str]]):
        results = []
        print(f"Predicting {len(sequences)} proteins...")
        for pid, seq in sequences:
            try:
                label, conf = self.predict_single(seq)
                print(f"{pid}: {label} ({conf:.3f})")
                results.append({
                    "id": pid,
                    "sequence_length": len(seq),
                    "prediction": label,
                    "confidence": conf
                })
            except Exception as e:
                print(f"Error processing {pid}: {e}")
                results.append({
                    "id": pid,
                    "sequence_length": len(seq),
                    "prediction": "ERROR",
                    "confidence": 0.0
                })
        return results


# --------------------------------------------------
# FASTA + SAVE
# --------------------------------------------------
def read_fasta(fasta_file: str) -> List[Tuple[str, str]]:
    seqs = []
    with open(fasta_file, 'r') as f:
        curr_id, curr_seq = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if curr_id:
                    seqs.append((curr_id, ''.join(curr_seq)))
                curr_id = line[1:].split()[0]
                curr_seq = []
            else:
                curr_seq.append(line)
        if curr_id:
            seqs.append((curr_id, ''.join(curr_seq)))
    return seqs


def save_predictions(results: List[dict], output_file: str):
    import csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'sequence_length', 'prediction', 'confidence'])
        writer.writeheader()
        writer.writerows(results)
    print(f"✓ Predictions saved to {output_file}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Predict heme-binding using Attention classifier')
    parser.add_argument('--model', type=str, default='best_attention_model.pt')
    parser.add_argument('--sequence', type=str, default=None)
    parser.add_argument('--fasta', type=str, default=None)
    parser.add_argument('--output', type=str, default='predictions.csv')
    parser.add_argument('--esm_model', type=str, default='esm2_t33_650M_UR50D')
    parser.add_argument('--max_len', type=int, default=256)
    args = parser.parse_args()

    if not args.sequence and not args.fasta:
        parser.error("Provide either --sequence or --fasta")

    print("=" * 70)
    print("HEME-BINDING PREDICTION (Attention Model)")
    print("=" * 70)

    predictor = HemePredictor(model_path=args.model, esm_model=args.esm_model, max_len=args.max_len)

    if args.sequence:
        label, conf = predictor.predict_single(args.sequence)
        print(f"\nPrediction: {label} (confidence={conf:.4f})")
    else:
        seqs = read_fasta(args.fasta)
        results = predictor.predict_batch(seqs)
        save_predictions(results, args.output)


if __name__ == "__main__":
    main()
