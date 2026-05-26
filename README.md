# Smart Health Monitor

Smart Health Monitor is a polished, public-ready README describing a small, reproducible project built to demonstrate data preprocessing, exploratory analysis, model development, and a safe inference API for educational review.

## How this repo maps to the assessment criteria

- **Data Preparation (what we used)**
  - Files: `src/preprocessing.py`, `src/config.py`
  - Libraries: pandas, numpy, scikit-learn
  - Key steps implemented: missing-value imputation (median), outlier clipping (1st/99th percentile), feature transforms (log1p for skew), derived feature (`age * bmi`), train/test split, and scaling with StandardScaler.

- **Exploratory Data Analysis (what we used)**
  - Files: `notebooks/health_analysis.ipynb`
  - Libraries: matplotlib, seaborn, pandas
  - Included: dataset summary, univariate/multivariate plots, boxplots for outlier detection, correlation matrix and short textual interpretations next to plots.

- **Modeling & Evaluation (what we used)**
  - Files: `src/model.py`, `src/data_loader.py`
  - Libraries: PyTorch
  - Model: compact LSTM-based network (LSTM -> linear -> sigmoid) adapted for tabular inputs (sequence length = 1)
  - Evaluation: designed for binary classification metrics (accuracy, precision, recall, F1) and ROC-AUC on a held-out test set.

- **Safety & Post-processing (what we used)**
  - File: `app.py` (prediction endpoint)
  - Behavior: after model scoring, apply rule-based checks for emergency conditions and adjust scores for obvious clinical signals (e.g., extreme heart rate or glucose), then return `risk_score` and categorical `status`.

- **Reproducibility & Documentation (what we used)**
  - Files: `notebooks/health_analysis.ipynb`, `requirements.txt`, `README.md`
  - Practice: notebook cells are annotated and runnable; preprocessing saves `data/processed/*` arrays so reviewers can load preprocessed data directly.

---

## Quick start (short commands)

1) Create a Python environment and install deps:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Run preprocessing (generates processed numpy arrays):
```python
from src.preprocessing import full_preprocessing_and_save
full_preprocessing_and_save()
```

3) Start the demo API and test a sample payload:
```powershell
python app.py
# then POST to http://127.0.0.1:5000/predict with a JSON body
```

---

## Privacy & responsible use

- This project is an educational prototype and is not clinically validated. Do not use it for clinical decision-making.

## Contact & License

- MIT License — see `LICENSE`.
- For questions, open an issue or contact: https://github.com/yonasleykun27



