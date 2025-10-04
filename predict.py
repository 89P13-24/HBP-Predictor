"""
Script 4: Predict Heme-Binding for New Proteins
================================================
Makes predictions for new protein sequences.

Installation:
    pip install torch fair-esm

Usage:
    # Single sequence
    python predict.py --sequence MKALIVLGL...
    
    # From FASTA file
    python predict.py --fasta new_proteins.fasta
    
    # Batch prediction
    python predict.py --fasta proteins.fasta --output predictions.csv

Output:
    Predictions with confidence scores
"""

import torch
import torch.nn.functional as F
import argparse
from typing import List, Tuple


class AttentionHemeClassifier(torch.nn.Module):
    """Same model architecture as training"""
    
    def __init__(self, input_dim=1280, hidden_dim=512, num_heads=8, dropout=0.3):
        super().__init__()
        
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        self.classifier = torch.nn.Linear(hidden_dim // 2, 2)
        self.ln1 = torch.nn.LayerNorm(input_dim)
        self.ln2 = torch.nn.LayerNorm(hidden_dim // 2)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.ln1(x + attn_out)
        x = x.squeeze(1)
        x = self.ffn(x)
        x = self.ln2(x)
        logits = self.classifier(x)
        return logits, attn_weights


class HemePredictor:
    """Prediction pipeline"""
    
    def __init__(self, model_path='best_heme_model.pt', esm_model='esm2_t33_650M_UR50D'):
        """Load trained model and ESM-2"""
        
        print("Loading models...")
        
        # Load ESM-2
        try:
            import esm
            if esm_model == "esm2_t33_650M_UR50D":
                self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                self.repr_layer = 33
            elif esm_model == "esm2_t12_35M_UR50D":
                self.esm_model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
                self.repr_layer = 12
            else:
                raise ValueError(f"Unsupported model: {esm_model}")
            
            self.batch_converter = self.alphabet.get_batch_converter()
            self.esm_model.eval()
        except ImportError:
            print("ERROR: fair-esm not installed. Run: pip install fair-esm")
            raise
        
        # Load classifier
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        checkpoint = torch.load(model_path, map_location=self.device)
        embedding_dim = 1280  # ESM-2 embedding dimension
        
        self.classifier = AttentionHemeClassifier(input_dim=embedding_dim)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()
        
        self.esm_model = self.esm_model.to(self.device)
        self.classifier = self.classifier.to(self.device)
        
        print(f"✓ Models loaded on {self.device}")
    
    def extract_embedding(self, sequence: str) -> torch.Tensor:
        """Extract ESM-2 embedding for a single sequence"""
        with torch.no_grad():
            data = [("protein", sequence)]
            _, _, tokens = self.batch_converter(data)
            tokens = tokens.to(self.device)
            
            results = self.esm_model(tokens, repr_layers=[self.repr_layer])
            token_repr = results["representations"][self.repr_layer]
            
            # Mean pooling
            seq_len = len(sequence)
            embedding = token_repr[0, 1:seq_len+1].mean(0)
            
        return embedding
    
    def predict_single(self, sequence: str) -> Tuple[str, float]:
        """
        Predict for a single sequence
        
        Returns:
            prediction: 'Heme-binding' or 'Non-heme-binding'
            confidence: probability score
        """
        # Get embedding
        embedding = self.extract_embedding(sequence)
        embedding = embedding.unsqueeze(0)  # Add batch dimension
        
        # Predict
        with torch.no_grad():
            logits, _ = self.classifier(embedding)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        prediction = 'Heme-binding' if pred_class == 1 else 'Non-heme-binding'
        
        return prediction, confidence
    
    def predict_batch(self, sequences: List[Tuple[str, str]]) -> List[dict]:
        """
        Predict for multiple sequences
        
        Args:
            sequences: List of (id, sequence) tuples
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        print(f"Predicting for {len(sequences)} proteins...")
        
        for protein_id, sequence in sequences:
            try:
                prediction, confidence = self.predict_single(sequence)
                results.append({
                    'id': protein_id,
                    'sequence_length': len(sequence),
                    'prediction': prediction,
                    'confidence': confidence
                })
                print(f"{protein_id}: {prediction} ({confidence:.4f})")
            except Exception as e:
                print(f"Error processing {protein_id}: {e}")
                results.append({
                    'id': protein_id,
                    'sequence_length': len(sequence),
                    'prediction': 'ERROR',
                    'confidence': 0.0
                })
        
        return results


def read_fasta(fasta_file: str) -> List[Tuple[str, str]]:
    """Read sequences from FASTA file"""
    sequences = []
    with open(fasta_file, 'r') as f:
        current_id = None
        current_seq = []
        
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences.append((current_id, ''.join(current_seq)))
                current_id = line[1:].split()[0]  # Get first part of header
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id:
            sequences.append((current_id, ''.join(current_seq)))
    
    return sequences


def save_predictions(results: List[dict], output_file: str):
    """Save predictions to CSV"""
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'sequence_length', 'prediction', 'confidence'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Predictions saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Predict heme-binding for protein sequences')
    parser.add_argument('--model', type=str, default='best_heme_model.pt',
                       help='Path to trained model')
    parser.add_argument('--sequence', type=str, default=None,
                       help='Single protein sequence')
    parser.add_argument('--fasta', type=str, default=None,
                       help='FASTA file with sequences')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file')
    parser.add_argument('--esm_model', type=str, default='esm2_t33_650M_UR50D',
                       help='ESM-2 model to use')
    
    args = parser.parse_args()
    
    if not args.sequence and not args.fasta:
        parser.error("Provide either --sequence or --fasta")
    
    print("="*70)
    print("HEME-BINDING PREDICTION")
    print("="*70)
    
    # Load predictor
    predictor = HemePredictor(model_path=args.model, esm_model=args.esm_model)
    
    # Single sequence prediction
    if args.sequence:
        print(f"\nPredicting for sequence (length: {len(args.sequence)})...")
        prediction, confidence = predictor.predict_single(args.sequence)
        
        print("\n" + "="*70)
        print("RESULT")
        print("="*70)
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        
        if prediction == 'Heme-binding':
            print("\n✓ This protein is predicted to bind heme")
        else:
            print("\n✗ This protein is predicted NOT to bind heme")
    
    # Batch prediction from FASTA
    elif args.fasta:
        sequences = read_fasta(args.fasta)
        print(f"\nLoaded {len(sequences)} sequences from {args.fasta}")
        
        results = predictor.predict_batch(sequences)
        
        # Summary
        heme_binding = sum(1 for r in results if r['prediction'] == 'Heme-binding')
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total proteins: {len(results)}")
        print(f"Heme-binding: {heme_binding} ({heme_binding/len(results)*100:.1f}%)")
        print(f"Non-heme-binding: {len(results)-heme_binding}")
        
        # Save results
        save_predictions(results, args.output)


if __name__ == "__main__":
    main()