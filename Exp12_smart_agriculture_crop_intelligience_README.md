# AgriSense — Smart Agriculture & Crop Intelligence

An interactive Streamlit dashboard combining the core concepts from 11 machine-learning lab experiments in a single agriculture-themed project.

## Features

- **Experiment 01 — EDA:** Pandas summaries, missing-value inspection, Matplotlib and Seaborn plots, correlation heatmap, CSV upload/download.
- **Experiments 02–03 — Regression:** Simple and multiple linear regression with MAE, MSE, RMSE, R², and actual-vs-predicted visualization.
- **Experiments 04–08 — Classification:** Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes; metrics, confusion matrix, classification report.
- **Scaling practical:** Compare original features, Min-Max normalization, and standardization.
- **Experiment 09 — Linear SVM:** Linear kernel demonstration.
- **Experiment 10 — Non-linear SVM:** Polynomial and RBF kernels, with adjustable C and gamma.
- **Experiment 11 — Decision Tree vs Random Forest:** Separate classification and regression workflows; shared train/test split; comparison metrics; confusion matrix / actual-vs-predicted plot; feature importance; downloadable outputs.
- **Crop Intelligence Playground:** Interactive inputs and an illustrative crop-condition prediction.

## Run locally

### macOS / Linux
```bash
cd Smart_Agriculture_Crop_Intelligence
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Windows PowerShell
```powershell
cd Smart_Agriculture_Crop_Intelligence
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Dataset notes

The default AgriSense dataset is generated in code with a fixed random seed, so no external data file is required to start. It includes rainfall, temperature, humidity, soil pH, N/P/K-style synthetic values, farm area, synthetic yield, and synthetic crop-condition labels. Small amounts of missingness are intentionally added for data-cleaning demonstrations.

The app also uses the Iris and Breast Cancer benchmark datasets from scikit-learn. The EDA page supports uploading a CSV.

**Important:** The crop dataset is synthetic and is not scientifically validated. It must not be used for real agricultural decisions. Experiment 09 may have been specified in your course as the UCI Banknote Authentication dataset; this combined app demonstrates linear SVM but does not bundle that exact external dataset. If the lab evaluator requires the exact dataset, download it separately and adapt the SVM page accordingly.

## Experiment 11 evaluation design

- Classification: AgriSense crop-condition labels or Iris.
- Regression: synthetic yield or farm area.
- Decision Tree and Random Forest receive the same selected features, split, and random seed.
- Classification metrics: accuracy, weighted precision, weighted recall, weighted F1; confusion matrix and classification report.
- Regression metrics: MAE, MSE, RMSE, R²; actual-vs-predicted plot.
- Feature importance is shown for fitted tree-based estimators.
- Metrics are intended for educational comparison, not claims about real-world farming performance.

## Project structure

```text
Smart_Agriculture_Crop_Intelligence/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
└── models/
```

## GitHub

```bash
git init
git add .
git commit -m "Build AgriSense smart agriculture ML dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Smart-Agriculture-Crop-Intelligence.git
git push -u origin main
```
