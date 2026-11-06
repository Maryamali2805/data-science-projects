# Data

This directory is **not tracked by git** (see root `.gitignore`).

## Source

**Dataset:** Credit Card Fraud Detection  
**URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
**Citation:** Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.

## How to download

### Option 1 — Kaggle CLI (recommended)

```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/ (never commit it)
python -m src.download_data
```

### Option 2 — Manual

1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click **Download** (requires free Kaggle account)
3. Unzip and place `creditcard.csv` in `data/raw/`

## Expected layout after download

```
data/
  raw/
    creditcard.csv   (~143 MB, 284,807 rows × 31 columns)
```

## Columns

| Column | Description |
|--------|-------------|
| Time | Seconds elapsed between transaction and first transaction in dataset |
| V1–V28 | PCA-transformed features (original features anonymised) |
| Amount | Transaction amount in euros |
| Class | 0 = legitimate, 1 = fraudulent |
