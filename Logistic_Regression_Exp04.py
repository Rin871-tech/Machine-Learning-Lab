# -*- coding: utf-8 -*-
"""
Experiment No. 04
Implementation of Logistic Regression for Binary Classification

Dataset: Placement_Y_N.csv (CGPA, IQ -> placement)
Based on the professor-provided notebook, Part 2: Placement / Not Placed.

Install:
    python3 -m pip install -r requirements_logistic_regression.txt

Run:
    streamlit run Logistic_Regression_Exp04.py

Keep Placement_Y_N.csv in the same folder, or upload the CSV in the sidebar.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Experiment 04 | Logistic Regression",
    page_icon="🎓",
    layout="wide",
)

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
st.markdown(
    '<div class="main-title">🎓 Experiment No. 04: Logistic Regression</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Binary classification: predict student placement using CGPA and IQ</div>',
    unsafe_allow_html=True,
)

with st.expander("Aim, objectives and theory", expanded=False):
    st.markdown(
        r"""
**Aim:** To implement logistic regression for binary classification using Python and evaluate its performance.

**Objectives**
1. Study logistic regression for classification.
2. Implement logistic regression using gradient descent and scikit-learn.
3. Evaluate performance using accuracy, confusion matrix, precision, recall, F1-score, ROC and AUC.
4. Visualise the decision boundary for the two-feature dataset.

**Theory**

Logistic regression estimates the probability of a binary outcome using the sigmoid function:

\[
p(y=1\mid x)=\frac{1}{1+e^{-(w^T x+b)}}
\]

The default decision rule predicts class 1 when probability is at least 0.5, otherwise class 0.
The binary cross-entropy (log-loss) cost is:

\[
J(w,b)=-\frac{1}{m}\sum_{i=1}^{m}
\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]
\]

Accuracy is the fraction of correct predictions. Precision, recall and F1-score summarise class-level performance.
The ROC curve plots true-positive rate against false-positive rate at different thresholds; AUC summarises the curve.
"""
    )

with st.expander("Procedure followed (based on the professor's notebook)", expanded=False):
    st.markdown(
        """
1. Import NumPy, pandas, Matplotlib and scikit-learn.
2. Load the Placement_Y_N dataset and remove the unnecessary index column.
3. Check missing values and inspect the CGPA/IQ features and placement target.
4. Select CGPA and IQ as independent features and placement (0 = Not Placed, 1 = Placed) as target.
5. Split the data into training and testing sets, stratifying by class.
6. Scale features using a scaler fitted only on training data.
7. Train scikit-learn LogisticRegression and a logistic regression model implemented with gradient descent.
8. Predict classes and probabilities on the test set.
9. Evaluate accuracy, confusion matrix, classification report, ROC curve and AUC.
10. Visualise the CGPA/IQ scatter plot and the learned decision boundary.
11. Enter custom CGPA and IQ values to predict placement.
"""
    )


# ---------------------------------------------------------------------
# Dataset loading and validation
# ---------------------------------------------------------------------
def load_local_csv():
    candidates = [
        Path(__file__).resolve().parent / "Placement_Y_N.csv",
        Path.cwd() / "Placement_Y_N.csv",
        Path.cwd() / "Placement_Y_N (1).csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return pd.read_csv(candidate), candidate.name
            except Exception:
                continue
    return None, None


st.sidebar.header("⚙️ Dataset & model settings")
uploaded = st.sidebar.file_uploader("Upload placement CSV (optional)", type=["csv"])

if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
        source_name = uploaded.name
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
else:
    raw_df, source_name = load_local_csv()
    if raw_df is None:
        st.warning(
            "Place `Placement_Y_N.csv` beside this Python file or upload the dataset in the sidebar."
        )
        st.stop()

df = raw_df.copy()
df.columns = [str(col).strip() for col in df.columns]
df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed:")], errors="ignore")

required_columns = ["cgpa", "iq", "placement"]
missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    st.error(
        "The dataset must contain these columns: `cgpa`, `iq`, and `placement`. "
        f"Missing: {', '.join(missing_columns)}."
    )
    st.write("Columns found:", list(df.columns))
    st.stop()

for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df[required_columns].replace([np.inf, -np.inf], np.nan).dropna()
df["placement"] = df["placement"].astype(int)

if not set(df["placement"].unique()).issubset({0, 1}) or df["placement"].nunique() != 2:
    st.error("The `placement` target must contain both binary classes: 0 and 1.")
    st.stop()
if len(df) < 10:
    st.error("At least 10 complete rows are needed for this experiment.")
    st.stop()

st.sidebar.caption(f"Loaded dataset: {source_name}")
st.sidebar.subheader("Training settings")
test_percent = st.sidebar.slider("Test set (%)", 10, 40, 20, 5)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
threshold = st.sidebar.slider("Classification threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
learning_rate = st.sidebar.number_input(
    "Gradient descent learning rate", min_value=0.001, max_value=1.0,
    value=0.05, step=0.01, format="%.3f",
)
iterations = st.sidebar.slider("Gradient descent iterations", min_value=100, max_value=20000, value=5000, step=100)

features = ["cgpa", "iq"]
target = "placement"
X = df[features].to_numpy(dtype=float)
y = df[target].to_numpy(dtype=int)

if np.unique(X[:, 0]).size < 2 or np.unique(X[:, 1]).size < 2:
    st.error("Both CGPA and IQ must contain more than one distinct value.")
    st.stop()

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_percent / 100,
        random_state=int(random_state),
        stratify=y,
    )
except ValueError as exc:
    st.error(f"Unable to split the dataset: {exc}")
    st.stop()

# Fit scaling on training data only (as in the professor's notebook).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# Model 1: scikit-learn Logistic Regression
# ---------------------------------------------------------------------
sk_model = LogisticRegression(max_iter=2000)
sk_model.fit(X_train_scaled, y_train)
sk_prob = sk_model.predict_proba(X_test_scaled)[:, 1]
sk_pred = (sk_prob >= threshold).astype(int)

# ---------------------------------------------------------------------
# Model 2: Logistic Regression implemented using batch gradient descent
# ---------------------------------------------------------------------
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic_gradient_descent(X_scaled, y_values, rate, n_iterations):
    """Train weights and bias by minimizing binary cross-entropy."""
    n_samples, n_features = X_scaled.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    losses = []
    eps = 1e-12

    for _ in range(int(n_iterations)):
        logits = X_scaled @ weights + bias
        probabilities = sigmoid(logits)
        probabilities_safe = np.clip(probabilities, eps, 1.0 - eps)

        loss = -np.mean(
            y_values * np.log(probabilities_safe)
            + (1 - y_values) * np.log(1.0 - probabilities_safe)
        )
        losses.append(float(loss))

        error = probabilities - y_values
        grad_w = (X_scaled.T @ error) / n_samples
        grad_b = float(np.mean(error))
        weights -= rate * grad_w
        bias -= rate * grad_b

        if not np.all(np.isfinite(weights)) or not np.isfinite(bias):
            raise FloatingPointError(
                "Gradient descent became unstable. Reduce the learning rate in the sidebar."
            )

    return weights, bias, losses


try:
    gd_weights, gd_bias, gd_losses = train_logistic_gradient_descent(
        X_train_scaled, y_train.astype(float), float(learning_rate), int(iterations)
    )
except FloatingPointError as exc:
    st.error(str(exc))
    st.stop()

gd_prob = sigmoid(X_test_scaled @ gd_weights + gd_bias)
gd_pred = (gd_prob >= threshold).astype(int)

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def calculate_metrics(name, actual, predicted, probabilities):
    fpr, tpr, _ = roc_curve(actual, probabilities)
    return {
        "Model": name,
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1-score": f1_score(actual, predicted, zero_division=0),
        "AUC": auc(fpr, tpr),
    }


metrics_df = pd.DataFrame([
    calculate_metrics("scikit-learn LogisticRegression", y_test, sk_pred, sk_prob),
    calculate_metrics("Gradient Descent", y_test, gd_pred, gd_prob),
])

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.subheader("1. Dataset overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows loaded", f"{len(df):,}")
c2.metric("Training samples", f"{len(X_train):,}")
c3.metric("Testing samples", f"{len(X_test):,}")
c4.metric("Features", "2 (CGPA, IQ)")

with st.expander("View dataset and descriptive statistics", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.write("Missing values in the cleaned dataset")
    st.dataframe(df.isna().sum().rename("Missing values").to_frame(), use_container_width=True)
    st.write("Descriptive statistics")
    st.dataframe(df.describe().T, use_container_width=True)
    st.write("Class distribution (0 = Not Placed, 1 = Placed)")
    st.bar_chart(df["placement"].value_counts().sort_index())

st.subheader("2. Model performance on the test set")
st.caption(
    "Both models use the same stratified train/test split. The selected threshold is applied to class probabilities."
)
st.dataframe(
    metrics_df.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1-score": "{:.3f}",
        "AUC": "{:.3f}",
    }),
    use_container_width=True,
    hide_index=True,
)

best = metrics_df.loc[metrics_df["Accuracy"].idxmax()]
m1, m2, m3 = st.columns(3)
m1.metric("Best accuracy", best["Model"])
m2.metric("Accuracy", f"{best['Accuracy']:.1%}")
m3.metric("AUC", f"{best['AUC']:.3f}")

st.subheader("3. Confusion matrix and classification report")
model_choice = st.radio(
    "Choose model for detailed evaluation",
    ["scikit-learn LogisticRegression", "Gradient Descent"],
    horizontal=True,
)
if model_choice == "scikit-learn LogisticRegression":
    chosen_pred, chosen_prob = sk_pred, sk_prob
else:
    chosen_pred, chosen_prob = gd_pred, gd_prob

cm = confusion_matrix(y_test, chosen_pred, labels=[0, 1])
cm_fig, cm_ax = plt.subplots(figsize=(5.5, 4.2))
image = cm_ax.imshow(cm, interpolation="nearest")
cm_ax.set_title(f"Confusion Matrix — {model_choice}")
cm_ax.set_xlabel("Predicted label")
cm_ax.set_ylabel("True label")
cm_ax.set_xticks([0, 1], labels=["Not Placed (0)", "Placed (1)"], rotation=10)
cm_ax.set_yticks([0, 1], labels=["Not Placed (0)", "Placed (1)"])
for row in range(2):
    for col in range(2):
        cm_ax.text(col, row, str(cm[row, col]), ha="center", va="center")
cm_fig.colorbar(image, ax=cm_ax, fraction=0.046, pad=0.04)
cm_fig.tight_layout()
st.pyplot(cm_fig)
plt.close(cm_fig)

report = classification_report(
    y_test, chosen_pred, labels=[0, 1],
    target_names=["Not Placed (0)", "Placed (1)"],
    output_dict=True, zero_division=0
)
st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"), use_container_width=True)

st.subheader("4. ROC curve")
roc_fig, roc_ax = plt.subplots(figsize=(7, 5))
for name, probabilities in [
    ("scikit-learn LogisticRegression", sk_prob),
    ("Gradient Descent", gd_prob),
]:
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)
    roc_ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")
roc_ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
roc_ax.set_xlabel("False Positive Rate")
roc_ax.set_ylabel("True Positive Rate")
roc_ax.set_title("Receiver Operating Characteristic (ROC)")
roc_ax.grid(True, alpha=0.25)
roc_ax.legend()
roc_fig.tight_layout()
st.pyplot(roc_fig)
plt.close(roc_fig)

st.subheader("5. Decision boundary: CGPA vs IQ")
x_min, x_max = float(df["cgpa"].min() - 0.3), float(df["cgpa"].max() + 0.3)
y_min, y_max = float(df["iq"].min() - 8), float(df["iq"].max() + 8)
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 240), np.linspace(y_min, y_max, 240))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)

sk_grid_prob = sk_model.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
gd_grid_prob = sigmoid(grid_scaled @ gd_weights + gd_bias).reshape(xx.shape)
boundary_choice = st.selectbox(
    "Decision boundary model",
    ["scikit-learn LogisticRegression", "Gradient Descent"],
)
boundary_prob = sk_grid_prob if boundary_choice == "scikit-learn LogisticRegression" else gd_grid_prob

boundary_fig, boundary_ax = plt.subplots(figsize=(9, 6))
boundary_ax.contourf(xx, yy, boundary_prob, levels=[0, threshold, 1], alpha=0.18)
boundary_ax.contour(xx, yy, boundary_prob, levels=[threshold], linewidths=2)
scatter = boundary_ax.scatter(
    df["cgpa"], df["iq"], c=df["placement"], edgecolors="black", alpha=0.85
)
boundary_ax.set_xlabel("CGPA")
boundary_ax.set_ylabel("IQ")
boundary_ax.set_title(f"Decision Boundary — {boundary_choice} (threshold = {threshold:.2f})")
boundary_ax.grid(True, alpha=0.2)
legend_handles, _ = scatter.legend_elements()
boundary_ax.legend(legend_handles, ["Not Placed (0)", "Placed (1)"], title="Actual class")
boundary_fig.tight_layout()
st.pyplot(boundary_fig)
plt.close(boundary_fig)

with st.expander("Gradient descent learning curve", expanded=False):
    loss_fig, loss_ax = plt.subplots(figsize=(8, 4))
    loss_ax.plot(np.arange(1, len(gd_losses) + 1), gd_losses)
    loss_ax.set_xlabel("Iteration")
    loss_ax.set_ylabel("Binary cross-entropy loss")
    loss_ax.set_title("Gradient Descent Convergence")
    loss_ax.grid(True, alpha=0.25)
    loss_fig.tight_layout()
    st.pyplot(loss_fig)
    plt.close(loss_fig)
    st.write(f"Final training log loss: {gd_losses[-1]:.6f}")

st.subheader("6. Predict placement for a new student")
st.write("Enter CGPA and IQ to estimate placement probability.")
with st.form("student_prediction"):
    col_a, col_b = st.columns(2)
    with col_a:
        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.8,
            step=0.1,
            format="%.2f",
        )
    with col_b:
        iq = st.number_input(
            "IQ",
            min_value=1.0,
            max_value=300.0,
            value=120.0,
            step=1.0,
            format="%.1f",
        )
    prediction_model = st.selectbox(
        "Prediction model",
        ["scikit-learn LogisticRegression", "Gradient Descent"],
    )
    predict_button = st.form_submit_button("Predict placement", type="primary")

if predict_button:
    new_scaled = scaler.transform(np.array([[cgpa, iq]], dtype=float))
    if prediction_model == "scikit-learn LogisticRegression":
        probability = float(sk_model.predict_proba(new_scaled)[0, 1])
    else:
        probability = float(sigmoid(new_scaled @ gd_weights + gd_bias)[0])
    predicted_class = int(probability >= threshold)
    if predicted_class == 1:
        st.success(f"Likely to be PLACED — probability: {probability:.1%}")
    else:
        st.warning(f"Likely to be NOT PLACED — placement probability: {probability:.1%}")
    st.caption(f"Decision threshold: {threshold:.2f}. This is a model estimate, not a guarantee.")

st.subheader("7. Observations and conclusion")
selected_accuracy = accuracy_score(y_test, chosen_pred)
selected_fpr, selected_tpr, _ = roc_curve(y_test, chosen_prob)
selected_auc = auc(selected_fpr, selected_tpr)
st.markdown(
    f"""
**Observations**
- The cleaned dataset contains **{len(df)} rows** and two independent features: CGPA and IQ.
- The target is binary: **0 = Not Placed** and **1 = Placed**.
- The selected model (**{model_choice}**) achieved **{selected_accuracy:.1%} accuracy** and an AUC of **{selected_auc:.3f}** on the test split.
- The confusion matrix and classification report show correct and incorrect class predictions.
- The ROC curve shows the trade-off between true-positive and false-positive rates at different thresholds.

**Conclusion**

Logistic regression was implemented for binary classification using both a gradient-descent implementation
and scikit-learn. The models were evaluated using accuracy, confusion matrix, precision, recall, F1-score,
ROC curve and AUC. The decision-boundary plot visualises how CGPA and IQ relate to the predicted
placement class.
"""
)

st.caption("Machine Learning Lab Manual by Dr. Sachin Balawant Takmare | Experiment No. 04")
