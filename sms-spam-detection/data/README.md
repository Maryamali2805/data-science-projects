# Data

The raw data file is not committed to this repository.

## Source

**SMS Spam Collection** — UCI Machine Learning Repository, via Kaggle  
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

- ~5,572 SMS messages labelled ham or spam
- `spam.csv`: latin-1 encoded, columns `v1` (label), `v2` (text), plus three unnamed filler columns
- ~13% spam, ~87% ham

## How to get it

**Option A — automated (requires Kaggle API credentials):**

```bash
python -m src.download_data
```

Your `~/.kaggle/kaggle.json` must exist. Get it from https://www.kaggle.com/settings → API → Create New Token.

**Option B — manual:**

1. Download from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Extract `spam.csv`
3. Place it at `sms-spam-detection/data/raw/spam.csv`
