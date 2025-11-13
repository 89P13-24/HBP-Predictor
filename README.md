# Heme-Binding Protein Prediction Pipeline

Advanced machine learning pipeline using ESM-2 protein language models and attention mechanisms to predict whether proteins bind heme molecules.

## 📋 Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Steps](#pipeline-steps)
- [Script Details](#script-details)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works)
- 8GB+ RAM (16GB+ recommended)

### Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv heme_env
source heme_env/bin/activate  # On Windows: heme_env\Scripts\activate

# Install packages
pip install -r requirements.txt
```

## 🚀 Quick Start

### Option 1: Quick Test (300+300 proteins, ~1 hour)
```bash
# Step 1: Collect data
python collect_data.py --max_sequences 300 --max_length 500

# Step 2: Extract embeddings (takes ~20 min)
python extract_embeddings.py --batch_size 4

# Step 3: Train model (takes ~15 min)
python train.py --epochs 30

# Step 4: Make predictions
python predict.py --sequence "MKALIVLGLVLLSAALCGQAKDAENGAESAQVKGHGKKVVDALANAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
```

### Option 2: Production Run (full dataset, 3-4 hours)
```bash
# Step 1: Collect all available data
python collect_data.py --max_sequences 10000 --max_length 2000

# Step 2: Extract embeddings (takes ~2 hours)
python extract_embeddings.py --model esm2_t33_650M_UR50D --batch_size 4

# Step 3: Train model (takes ~1 hour)
python train_model.py --epochs 50 --lr 0.0001 --batch_size 32

# Step 4: Predict
python predict.py --fasta my_proteins.fasta --output results.csv
```

## 📊 Pipeline Steps

### Step 1: Data Collection (`collect_data.py`)
Fetches protein sequences from UniProt database.

**Arguments:**
- `--max_sequences`: Number of heme-binding proteins 
- `--max_length`: Maximum length of single protein sequence
- `--output`: Output file prefix

**Output:**
- `protein_sequences.pkl` - Pickle file with all data
- `sequences.fasta` - FASTA format

**Example:**
```bash
python collect_data.py --max_sequences 1000 --max_length 1000
```

### Step 2: Embedding Extraction (`extract_embeddings.py`)
Generates ESM-2 protein embeddings.

**Arguments:**
- `--input`: Input pickle file (default: protein_sequences.pkl)
- `--output`: Output file (default: protein_embeddings.pt)
- `--model`: ESM-2 model variant
  - `esm2_t12_35M_UR50D` - Fast (35M params)
  - `esm2_t33_650M_UR50D` - **Recommended** (650M params)
  - `esm2_t36_3B_UR50D` - Best quality (3B params, needs 16GB GPU)
- `--batch_size`: Batch size (reduce if OOM error)

**Output:**
- `protein_embeddings.pt` - PyTorch tensor with embeddings

**Example:**
```bash
# Standard
python extract_embeddings.py --batch_size 4

# If you have large GPU
python extract_embeddings.py --model esm2_t36_3B_UR50D --batch_size 8

# If running out of memory
python extract_embeddings.py --batch_size 2
```

### Step 3: Model Training (`train_model.py`)
Trains attention-based classifier.

**Arguments:**
- `--input`: Input embeddings file
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.0001)
- `--batch_size`: Batch size (default: 32)
- `--hidden_dim`: Hidden layer dimension (default: 512)
- `--patience`: Early stopping patience (default: 10)

**Output:**
- `best_heme_model.pt` - Best model checkpoint

**Example:**
```bash
python train_model.py --epochs 50 --lr 0.0001
```

### Step 4: Prediction (`predict.py`)
Make predictions for new proteins.

**Arguments:**
- `--model`: Path to trained model
- `--sequence`: Single protein sequence
- `--fasta`: FASTA file with multiple sequences
- `--output`: Output CSV file
- `--esm_model`: ESM-2 model (must match training)

**Output:**
- CSV file with predictions and confidence scores

**Examples:**
```bash
# Single sequence
python predict.py --sequence "MKALIVLGL..."

# Batch prediction
python predict.py --fasta new_proteins.fasta --output predictions.csv
```

## 📈 Performance

### Model Comparison

| ESM-2 Model | Params | Speed | Quality | GPU Memory |
|------------|--------|-------|---------|------------|
| esm2_t12_35M | 35M | Fast | Good | 4GB |
| esm2_t33_650M | 650M | Medium | **Best** | 8GB |
| esm2_t36_3B | 3B | Slow | Excellent | 16GB+ |

## 🛠️ Troubleshooting

### Out of Memory (OOM) Errors

**During embedding extraction:**
```bash
# Reduce batch size
python extract_embeddings.py --batch_size 2

# Or use smaller model
python extract_embeddings.py --model esm2_t12_35M_UR50D --batch_size 8
```

**During training:**
```bash
# Reduce batch size
python train_model.py --batch_size 16

# Reduce hidden dimension
python train_model.py --hidden_dim 256
```


## 📁 Output Files

After running the complete pipeline, you'll have:

```
project/
├── collect_data.py
├── extract_embeddings.py
├── train_model.py
├── predict.py
├── sequences.fasta                 # FASTA format
├── protein_embeddings.pt           # ESM-2 embeddings
├── best_heme_model.pt             # Trained model
├── training_history.pkl           # Training metrics
├── training_curves.png            # Visualization
└── predictions.csv                # Results
```

### Example Output
```
Protein: P12345
Prediction: Heme-binding
Confidence: 0.9547

This protein is predicted to bind heme with high confidence.
```

## 🧬 Example Proteins to Test

**Known heme-binding proteins:**
- Cytochrome c (UniProt: P99999)
- Myoglobin (UniProt: P02144)
- Hemoglobin (UniProt: P69905)

**Known non-heme proteins:**
- Lysozyme (UniProt: P00720)
- Insulin (UniProt: P01308)

## 📚 References

- **ESM-2**: Lin et al. (2023) "Evolutionary-scale prediction of atomic-level protein structure with a language model"
- **UniProt**: https://www.uniprot.org/
- **Heme proteins**: https://en.wikipedia.org/wiki/Heme_protein

## 📧 Need Help?

Common issues:
- Installation problems → Check Python version and CUDA compatibility
- Memory issues → Reduce batch size or use smaller model
- Low accuracy → Collect more data or use larger ESM-2 model
- Slow training → Use GPU if available
