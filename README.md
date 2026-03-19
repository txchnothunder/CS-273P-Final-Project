# Predictive Maintenance with Multi-Task Learning
CS273P Final Project — Ethan Wong

A PyTorch implementation of a Multi-Task Learning (MTL) architecture for predictive maintenance, trained on the AI4I 2020 dataset. The model shares a residual MLP backbone across two tasks:

- **Binary classification:** Did the machine fail?
- **6-class classification:** Which failure mode occurred? (TWF, HDF, PWF, OSF, RNF, or No Failure)

We trained both tasks jointly.
---

## Project Structure

```
CS-273P-Final-Project/
├── data/
│   └── ai4i2020.csv          # Dataset (download instructions below)
├── notebooks/
│   └── demo.ipynb            # End-to-end demo notebook
├── src/
│   ├── dataset.py            # Dataset, DataLoader, preprocessing
│   ├── model.py              # MTL architecture and loss function
│   ├── train.py              # Training loop
│   ├── test.py               # Test set evaluation
│   ├── evaluate.py           # Metrics, reports, confusion matrix
│   └── utils.py              # Seeding, checkpointing, plotting
├── checkpoints/              # Saved model weights (created at training)
├── results/                  # Saved plots (created at training)
├── smoke_test.py             # End-to-end sanity check
├── requirements.txt
└── setup.py
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/txchnothunder/CS-273P-Final-Project.git
cd CS-273P-Final-Project
```

### 2. Create and activate a conda environment

```bash
conda create -n cs273p-mtl python=3.10 -y
conda activate cs273p-mtl
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the AI4I 2020 Predictive Maintenance Dataset from the UCI ML Repository:

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Place the file at:
```
data/ai4i2020.csv
```

---

## Quickstart

Run the smoke test first to verify the environment is set up correctly:

```bash
python smoke_test.py
```

Expected output: `ALL CHECKS PASSED`

---

## Training

```bash
python src/train.py
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 50 | Maximum training epochs |
| `--lr` | 1e-3 | Adam learning rate |
| `--batch-size` | 64 | Samples per batch |
| `--alpha` | 0.5 | Weight on binary loss (1-alpha on type loss) |
| `--dropout` | 0.3 | Dropout rate |
| `--patience` | 10 | Early stopping patience |
| `--seed` | 42 | Random seed |
| `--no-early-stopping` | — | Run all epochs without early stopping |

Example with custom arguments:
```bash
python src/train.py --epochs 100 --lr 5e-4 --alpha 0.6
```

Outputs saved after training:
- `checkpoints/best_model.pt` — best model weights by validation loss
- `results/loss_curves.png` — train/val loss curves for total, binary, and type losses

---

## Evaluation

After training, run evaluation on the held-out test set:

```bash
python src/test.py
```

Optional arguments:
```bash
python src/test.py --checkpoint checkpoints/best_model.pt --results-dir results
```

Outputs:
- Classification report for both tasks printed to terminal
- `results/confusion_matrix.png` — raw and normalized confusion matrices for the 6-class task

---

## Demo Notebook

Open the notebook after training has completed:

```bash
jupyter notebook notebooks/demo.ipynb
```

Run all cells top to bottom. The notebook walks through dataset statistics, loads the saved checkpoint, evaluates on the test set, displays the loss curves and confusion matrix, and runs live inference on custom sensor readings. The last few cells contain a `predict_single()` function — edit the input values to test any machine configuration.

---

## Expected Results

After training on the default configuration (50 epochs, seed 42):

| Metric | Expected |
|---|---|
| Binary Recall | ~0.90+ |
| Binary F1 | ~0.35–0.45 |
| Binary ROC-AUC | ~0.90+ |
| Type F1 (macro) | ~0.35–0.45 |
| Type F1 (weighted) | ~0.88+ |

Binary precision will appear low (~0.23) due to the severe class imbalance — only 3.4% of samples are failures. The model is intentionally tuned toward high recall, since missing a real failure is a worse outcome than triggering an unnecessary inspection. RNF has only 19 samples in the full dataset and will typically score near 0 F1 regardless of model quality.

---

## Dependencies

| Package | Version |
|---|---|
| Python | >= 3.10 |
| PyTorch | >= 2.3.0 |
| scikit-learn | >= 1.4.0 |
| pandas | >= 2.2.0 |
| numpy | >= 1.26.0 |
| matplotlib | >= 3.8.0 |
| seaborn | >= 0.13.0 |
| tqdm | >= 4.66.0 |

Full list: `requirements.txt`

---

## Reproducing Results

All randomness is controlled through a single seed flag. To reproduce the reported results exactly:

```bash
python src/train.py --seed 42
python src/test.py --seed 42
```

The seed controls dataset splitting, weight initialization, and batch shuffling.