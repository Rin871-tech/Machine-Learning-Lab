# Experiment No. 11 — Decision Tree vs Random Forest

Two independent Streamlit applications:
1. `classification_app/app.py` — Student pass/fail classification using the UCI Student Performance dataset.
2. `regression_app/app.py` — California house-value regression using scikit-learn's California Housing dataset.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

The datasets are downloaded/loaded from their official sources the first time each app runs. Internet access is needed for the first run.

## Run classification UI
```bash
cd classification_app
streamlit run app.py
```

## Run regression UI
Open a second terminal:
```bash
cd regression_app
streamlit run app.py
```

## Methodology
- Both models use the same 80/20 train-test split for each task.
- Fixed random seed for reproducibility.
- Classification excludes G1, G2, and G3 from input features to avoid target leakage.
- Classification target: pass if G3 >= 10, otherwise fail.
- Classification metrics: accuracy, precision, recall, F1-score, confusion matrix.
- Regression metrics: MAE, RMSE, R².
- Feature importance is shown for both models.
- Decision tree depth is limited to help control overfitting; Random Forest uses multiple trees.

## Dataset sources
- UCI Student Performance: https://archive.ics.uci.edu/dataset/320/student+performance
- California Housing docs: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

## Academic note
These are demonstration models, not systems for consequential student decisions or real-time property valuations.
