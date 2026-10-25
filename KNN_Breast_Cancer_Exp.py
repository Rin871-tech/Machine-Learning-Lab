# -*- coding: utf-8 -*-
"""
Experiment: Implementation of K-Nearest Neighbors (KNN) Classifier
Dataset: Breast Cancer Wisconsin Diagnostic dataset
Based on the professor-provided KNN_Using_breast_cancer_dataset.ipynb.

Install:
    python3 -m pip install -r requirements_knn.txt

Run:
    streamlit run KNN_Breast_Cancer_Exp.py

Keep breast_cancer_knn.csv beside this file or upload it in the sidebar.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="KNN Classifier | Breast Cancer", page_icon="🔎", layout="wide")

st.markdown(
    """
    <style>
      .main-title {font-size: 2rem; font-weight: 750; margin-bottom: .2rem;}
      .subtitle {color: #64748b; margin-bottom: 1rem;}
      div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22);
        padding: .8rem; border-radius: .7rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="main-title">🔎 K-Nearest Neighbors (KNN) Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Experiment: Breast Cancer Classification using KNN</div>', unsafe_allow_html=True)

with st.expander("Aim, objectives and theory", expanded=False):
    st.markdown(
        r"""
**Aim:** To implement the K-Nearest Neighbors (KNN) classifier using Python and evaluate its performance on a classification dataset.

**Objectives**
1. Study the concept of KNN.
2. Implement KNN using scikit-learn.
3. Evaluate performance using accuracy, confusion matrix, and classification report.
4. Visualise decision boundaries for KNN.

**Theory**

KNN is an instance-based algorithm. For a new sample, it calculates distances to training samples,
selects the \(k\) nearest neighbors, and predicts the majority class.

- **k:** number of neighbors.
- **Distance metric:** Euclidean, Manhattan, or Minkowski.
- **Weights:** uniform voting or weighting neighbors by inverse distance.
- **Standardization:** important because KNN relies on distances and the features have different scales.
"""
    )

with st.expander("Procedure followed (based on the professor's notebook)", expanded=False):
    st.markdown(
        """
1. Load the Breast Cancer Wisconsin Diagnostic dataset.
2. Remove the unnecessary `id` column.
3. Use `diagnosis` as the target and the 30 numeric measurements as features.
4. Split the data using `test_size=0.2` and `random_state=2`.
5. Standardize features by fitting `StandardScaler` on training data only.
6. Train KNN and compare k values from 1 to 15.
7. Evaluate accuracy, confusion matrix, and classification report.
8. Visualise a two-feature decision boundary.
9. Enter feature values to obtain an illustrative model prediction.
"""
    )

def load_local_csv():
    candidates = [
        Path(__file__).resolve().parent / "breast_cancer_knn.csv",
        Path.cwd() / "breast_cancer_knn.csv",
        Path.cwd() / "breast-cancer.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                return pd.read_csv(path), path.name
            except Exception:
                pass
    return None, None

st.sidebar.header("⚙️ Dataset & KNN settings")
uploaded = st.sidebar.file_uploader("Upload breast cancer CSV (optional)", type=["csv"])
if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
        source_name = uploaded.name
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    raw_df, source_name = load_local_csv()
    if raw_df is None:
        st.warning("Keep `breast_cancer_knn.csv` beside this Python file or upload the dataset in the sidebar.")
        st.stop()

raw_df.columns = [str(c).strip() for c in raw_df.columns]
raw_df = raw_df.drop(columns=[c for c in raw_df.columns if c.lower().startswith("unnamed:")], errors="ignore")
if "diagnosis" not in raw_df.columns:
    st.error("Dataset must contain a `diagnosis` target column.")
    st.write("Columns found:", list(raw_df.columns))
    st.stop()

# Drop ID if present and use numeric measurements as features.
raw_df = raw_df.drop(columns=["id", "ID"], errors="ignore")
feature_cols = [c for c in raw_df.columns if c != "diagnosis"]
for c in feature_cols:
    raw_df[c] = pd.to_numeric(raw_df[c], errors="coerce")
raw_df["diagnosis"] = raw_df["diagnosis"].astype(str).str.strip().str.upper()
data = raw_df[["diagnosis"] + feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
if data["diagnosis"].nunique() != 2:
    st.error("The diagnosis column must contain two classes (commonly B and M).")
    st.stop()
if len(data) < 20:
    st.error("At least 20 complete rows are required.")
    st.stop()

# Map common source labels to readable class names while preserving class encoding.
unique_labels = sorted(data["diagnosis"].unique().tolist())
if set(unique_labels) == {"B", "M"}:
    label_names = {"B": "Benign (B)", "M": "Malignant (M)"}
else:
    label_names = {label: str(label) for label in unique_labels}

st.sidebar.caption(f"Dataset: {source_name} · {len(data)} complete rows")
test_size = st.sidebar.slider("Test set (%)", 10, 40, 20, 5)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=2, step=1)
k = st.sidebar.slider("Number of neighbors (k)", min_value=1, max_value=15, value=3, step=1)
metric = st.sidebar.selectbox("Distance metric", ["minkowski", "euclidean", "manhattan"], index=0)
p_value = 1 if metric == "manhattan" else 2
weights = st.sidebar.selectbox("Neighbor weights", ["uniform", "distance"], index=0)

X = data[feature_cols].astype(float)
y = data["diagnosis"].astype(str)
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=int(random_state), stratify=y
    )
except ValueError as exc:
    st.error(f"Could not split the dataset: {exc}")
    st.stop()

# Pipeline prevents scaling leakage: scaler fits only on training features.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=k, metric=metric, p=p_value, weights=weights)),
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob_all = model.predict_proba(X_test)
classes = list(model.named_steps["knn"].classes_)
positive_label = "M" if "M" in classes else classes[-1]
positive_index = classes.index(positive_label)
y_prob = y_prob_all[:, positive_index]
accuracy = accuracy_score(y_test, y_pred)

st.subheader("1. Dataset overview")
a, b, c, d = st.columns(4)
a.metric("Rows loaded", f"{len(data):,}")
b.metric("Input features", len(feature_cols))
c.metric("Training samples", f"{len(X_train):,}")
d.metric("Testing samples", f"{len(X_test):,}")

with st.expander("View dataset and descriptive statistics", expanded=False):
    st.dataframe(data.head(20), use_container_width=True)
    st.write("Class distribution")
    st.dataframe(data["diagnosis"].map(label_names).value_counts().to_frame("Count"))
    st.write("Feature statistics")
    st.dataframe(X.describe().T, use_container_width=True)

st.subheader("2. KNN model evaluation")
m1, m2, m3 = st.columns(3)
m1.metric("Test accuracy", f"{accuracy:.1%}")
m2.metric("k neighbors", k)
m3.metric("Distance metric", "Manhattan" if metric == "manhattan" else ("Euclidean" if metric == "euclidean" else "Minkowski (p=2)"))
st.caption(f"Train/test split: {100-test_size}% / {test_size}%; random state: {int(random_state)}; weights: {weights}.")

st.subheader("3. Confusion matrix")
labels_order = classes
cm = confusion_matrix(y_test, y_pred, labels=labels_order)
cm_fig, cm_ax = plt.subplots(figsize=(6, 4.5))
im = cm_ax.imshow(cm, interpolation="nearest")
cm_ax.set_title("Confusion Matrix — KNN")
cm_ax.set_xlabel("Predicted label")
cm_ax.set_ylabel("Actual label")
cm_ax.set_xticks(range(len(labels_order)), labels=[label_names[x] for x in labels_order], rotation=10)
cm_ax.set_yticks(range(len(labels_order)), labels=[label_names[x] for x in labels_order])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        cm_ax.text(j, i, str(cm[i, j]), ha="center", va="center")
cm_fig.colorbar(im, ax=cm_ax, fraction=0.046, pad=0.04)
cm_fig.tight_layout()
st.pyplot(cm_fig)
plt.close(cm_fig)

st.subheader("4. Classification report")
report = classification_report(
    y_test, y_pred, labels=labels_order,
    target_names=[label_names[x] for x in labels_order],
    output_dict=True, zero_division=0,
)
st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"), use_container_width=True)

st.subheader("5. Compare accuracy for k = 1 to 15")
k_scores = []
for candidate_k in range(1, 16):
    candidate = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=candidate_k, metric=metric, p=p_value, weights=weights
        )),
    ])
    candidate.fit(X_train, y_train)
    k_scores.append(accuracy_score(y_test, candidate.predict(X_test)))
scores_df = pd.DataFrame({"k": list(range(1, 16)), "Test accuracy": k_scores})
best_index = int(np.argmax(k_scores))
best_k = int(scores_df.iloc[best_index]["k"])
best_acc = float(scores_df.iloc[best_index]["Test accuracy"])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(scores_df["k"], scores_df["Test accuracy"], marker="o")
ax.axvline(k, linestyle="--", label=f"Current k = {k}")
ax.axvline(best_k, linestyle=":", label=f"Best test k = {best_k}")
ax.set_xlabel("Number of neighbors (k)")
ax.set_ylabel("Test accuracy")
ax.set_title("KNN Accuracy vs k")
ax.set_xticks(range(1, 16))
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption(f"Best accuracy among k=1…15 on this test split: {best_acc:.1%} at k={best_k}. For rigorous tuning, choose k using cross-validation on training data rather than selecting it from the test set.")

st.subheader("6. Decision boundary (two selected features)")
st.caption("The main evaluation above uses all 30 features. This plot fits a separate 2D KNN model using only the two selected features for visualisation.")
feature_options = feature_cols
default_x = "radius_mean" if "radius_mean" in feature_options else feature_options[0]
default_y = "texture_mean" if "texture_mean" in feature_options else feature_options[1]
left, right = st.columns(2)
with left:
    x_feature = st.selectbox("X-axis feature", feature_options, index=feature_options.index(default_x))
with right:
    y_choices = [f for f in feature_options if f != x_feature]
    y_feature = st.selectbox("Y-axis feature", y_choices, index=y_choices.index(default_y if default_y in y_choices else y_choices[0]))

X_2d = data[[x_feature, y_feature]]
boundary_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=k, metric=metric, p=p_value, weights=weights)),
])
boundary_model.fit(X_2d, y)
x_min, x_max = float(X_2d[x_feature].min()), float(X_2d[x_feature].max())
y_min, y_max = float(X_2d[y_feature].min()), float(X_2d[y_feature].max())
x_pad = max((x_max - x_min) * 0.06, 0.01)
y_pad = max((y_max - y_min) * 0.06, 0.01)
xx, yy = np.meshgrid(
    np.linspace(x_min - x_pad, x_max + x_pad, 220),
    np.linspace(y_min - y_pad, y_max + y_pad, 220),
)
grid = pd.DataFrame({x_feature: xx.ravel(), y_feature: yy.ravel()})
zz = boundary_model.predict(grid)
class_to_int = {label: idx for idx, label in enumerate(labels_order)}
zz_num = np.array([class_to_int[v] for v in zz]).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(9, 6))
ax.contourf(xx, yy, zz_num, alpha=0.22, levels=np.arange(-0.5, len(labels_order) + 0.5, 1))
for label in labels_order:
    subset = data[data["diagnosis"] == label]
    ax.scatter(subset[x_feature], subset[y_feature], label=label_names[label], alpha=0.8, edgecolors="black", linewidths=0.25)
ax.set_xlabel(x_feature)
ax.set_ylabel(y_feature)
ax.set_title(f"2D KNN Decision Boundary (k={k})")
ax.grid(True, alpha=0.15)
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("7. Predict diagnosis for a new sample")
st.warning("Educational demonstration only. This classifier is not a medical diagnostic tool and must not be used for healthcare decisions.")
medians = X_train.median()
mins = X_train.min()
maxs = X_train.max()
with st.form("knn_prediction_form"):
    values = {}
    left_col, right_col = st.columns(2)
    for i, feature in enumerate(feature_cols):
        container = left_col if i % 2 == 0 else right_col
        with container:
            values[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=float(mins[feature]),
                max_value=float(maxs[feature]),
                value=float(medians[feature]),
                format="%.5f",
                help=f"Training range: {mins[feature]:.5f} to {maxs[feature]:.5f}",
            )
    submitted = st.form_submit_button("Predict diagnosis", type="primary")
if submitted:
    sample = pd.DataFrame([values], columns=feature_cols)
    predicted = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    prob_map = {label_names[c]: float(probabilities[i]) for i, c in enumerate(classes)}
    if predicted == "M":
        st.error(f"Model prediction: **Malignant (M)** — estimated class probability {prob_map.get(label_names[predicted], 0):.1%}")
    elif predicted == "B":
        st.success(f"Model prediction: **Benign (B)** — estimated class probability {prob_map.get(label_names[predicted], 0):.1%}")
    else:
        st.info(f"Predicted class: **{label_names[predicted]}**")
    st.dataframe(pd.DataFrame({"Class": list(prob_map.keys()), "Model probability": list(prob_map.values())}).style.format({"Model probability": "{:.1%}"}), hide_index=True, use_container_width=True)
    st.caption("Probabilities are model outputs, not clinical certainty.")

with st.expander("View test-set predictions", expanded=False):
    pred_df = X_test.copy()
    pred_df["Actual diagnosis"] = y_test.map(label_names)
    pred_df["Predicted diagnosis"] = pd.Series(y_pred, index=y_test.index).map(label_names)
    pred_df["Correct"] = pred_df["Actual diagnosis"] == pred_df["Predicted diagnosis"]
    st.dataframe(pred_df, use_container_width=True)
    st.download_button(
        "Download test predictions as CSV",
        data=pred_df.to_csv(index=True).encode("utf-8"),
        file_name="knn_breast_cancer_predictions.csv",
        mime="text/csv",
    )

st.subheader("8. Observations and conclusion")
st.markdown(
    f"""
**Observations**
- The dataset contains **{len(data)} records** and **{len(feature_cols)} numeric measurement features**.
- KNN uses distance between standardized feature vectors to find neighboring training samples.
- The selected model achieved **{accuracy:.1%} test accuracy** with k = **{k}**.
- Accuracy varies with k, distance metric, and neighbor weighting.
- A smaller k can create a more flexible boundary and may overfit; a larger k generally smooths the boundary.

**Conclusion**

The K-Nearest Neighbors classifier was implemented using scikit-learn and the breast cancer dataset.
Features were standardized before distance-based classification. The model was evaluated using accuracy,
confusion matrix, and classification report, and a two-feature decision boundary was plotted to visualise
how KNN separates the two diagnosis classes.
"""
)
st.caption("Machine Learning Lab Manual by Dr. Sachin Balawant Takmare | KNN Classifier")
