# -*- coding: utf-8 -*-
"""
Experiment No. 06 — Decision Tree Classification (Gini Index)
Dataset: UCI Heart Disease Cleveland subset, extracted from the user-provided markdown.
Based on the professor-provided Decision_Tree_Gini_HeartDisease.ipynb.

Install:
    python3 -m pip install -r requirements_decision_tree.txt

Run:
    streamlit run Decision_Tree_Exp06.py

Keep heart_disease.csv in the same folder or upload a CSV in the sidebar.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

st.set_page_config(page_title="Experiment 06 | Decision Tree", page_icon="🌳", layout="wide")

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
    '<div class="main-title">🌳 Experiment No. 06: Decision Tree Classifier</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Gini Index · Heart Disease Dataset · Model evaluation and pruning</div>',
    unsafe_allow_html=True,
)

with st.expander("Aim, objectives and theory", expanded=False):
    st.markdown(
        r"""
**Aim:** To implement a Decision Tree classifier using Python and evaluate its performance on a given dataset.

**Objectives**
1. Understand the concept and working principle of Decision Tree algorithms.
2. Build and visualise a Decision Tree using scikit-learn.
3. Analyse the decision-making process based on feature splits.
4. Evaluate model performance using standard classification metrics.
5. Understand overfitting and pruning in Decision Trees.

**Theory**

A Decision Tree is a supervised learning algorithm that recursively splits data into subsets.
Internal nodes represent feature tests, branches represent outcomes of tests, and leaves represent predicted classes.

The Gini impurity is:

\[
Gini = 1 - \sum_{i=1}^{c} p_i^2
\]

where \(p_i\) is the proportion of class \(i\) in a node. A pure node has Gini impurity 0.
The CART algorithm selects splits that reduce the weighted impurity of the child nodes.
Tree depth, minimum samples per leaf, and pruning can help control overfitting.
"""
    )

with st.expander("Procedure followed (based on the professor's notebook)", expanded=False):
    st.markdown(
        """
1. Import pandas, NumPy, Matplotlib and scikit-learn.
2. Load the Heart Disease dataset.
3. Separate the 13 input features from the binary target.
4. Split data into training and testing sets using an 80:20 stratified split.
5. Train a DecisionTreeClassifier with the Gini criterion.
6. Predict test-set labels and probabilities.
7. Evaluate accuracy, confusion matrix, classification report and ROC-AUC.
8. Visualise the tree, extract decision rules and rank feature importances.
9. Compare training and testing accuracy at different maximum depths to study overfitting.
10. Enter feature values to obtain an illustrative model prediction.
"""
    )

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (encoded 0–3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting ECG result (encoded)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Number of major vessels (0–4)",
    "thal": "Thalassemia (encoded)",
}
TARGET_NAMES = ["No Disease", "Disease"]
FEATURES = list(FEATURE_DESCRIPTIONS.keys())


def load_local_csv():
    candidates = [
        Path(__file__).resolve().parent / "heart_disease.csv",
        Path.cwd() / "heart_disease.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                return pd.read_csv(path), path.name
            except Exception:
                pass
    return None, None


st.sidebar.header("⚙️ Dataset & model settings")
uploaded = st.sidebar.file_uploader("Upload Heart Disease CSV (optional)", type=["csv"])
if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
        source_name = uploaded.name
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        st.stop()
else:
    raw_df, source_name = load_local_csv()
    if raw_df is None:
        st.warning("Keep `heart_disease.csv` beside this Python file or upload the dataset in the sidebar.")
        st.stop()

raw_df.columns = [str(c).strip() for c in raw_df.columns]
raw_df = raw_df.drop(
    columns=[c for c in raw_df.columns if c.lower().startswith("unnamed:")],
    errors="ignore",
)
required = FEATURES + ["target"]
missing = [c for c in required if c not in raw_df.columns]
if missing:
    st.error("Dataset is missing required columns: " + ", ".join(missing))
    st.write("Columns found:", list(raw_df.columns))
    st.stop()

df = raw_df[required].copy()
for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan).dropna()
if not set(df["target"].unique()).issubset({0, 1}) or df["target"].nunique() != 2:
    st.error("The target column must contain both classes 0 and 1.")
    st.stop()
df["target"] = df["target"].astype(int)
if len(df) < 20:
    st.error("At least 20 complete rows are needed to train and evaluate the tree.")
    st.stop()

st.sidebar.caption(f"Dataset: {source_name} · {len(df)} complete rows")
st.sidebar.subheader("Tree configuration")
max_depth = st.sidebar.slider("Maximum tree depth", min_value=1, max_value=12, value=4)
min_samples_split = st.sidebar.slider("Minimum samples to split", min_value=2, max_value=30, value=10)
min_samples_leaf = st.sidebar.slider("Minimum samples per leaf", min_value=1, max_value=20, value=5)
criterion = st.sidebar.selectbox("Splitting criterion", ["gini", "entropy", "log_loss"], index=0)
test_size = st.sidebar.slider("Test set (%)", min_value=10, max_value=40, value=20, step=5)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

X = df[FEATURES]
y = df["target"]
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=int(random_state), stratify=y
    )
except ValueError as exc:
    st.error(f"Could not split dataset: {exc}")
    st.stop()

model = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=int(random_state),
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob)

st.subheader("1. Dataset overview")
a, b, c, d = st.columns(4)
a.metric("Rows used", f"{len(df):,}")
b.metric("Input features", len(FEATURES))
c.metric("Training samples", f"{len(X_train):,}")
d.metric("Testing samples", f"{len(X_test):,}")

with st.expander("View dataset, feature descriptions and statistics", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.write("Feature descriptions")
    st.dataframe(
        pd.DataFrame({"Feature": FEATURES, "Description": [FEATURE_DESCRIPTIONS[f] for f in FEATURES]}),
        use_container_width=True,
        hide_index=True,
    )
    st.write("Descriptive statistics")
    st.dataframe(df.describe().T, use_container_width=True)
    st.write("Target class counts (0 = No Disease, 1 = Disease)")
    st.dataframe(df["target"].map({0: "No Disease", 1: "Disease"}).value_counts().to_frame("Count"))

st.subheader("2. Model evaluation")
m1, m2, m3 = st.columns(3)
m1.metric("Test accuracy", f"{accuracy:.1%}")
m2.metric("ROC-AUC", f"{auc_score:.3f}")
m3.metric("Tree depth", model.get_depth())
st.caption(
    f"Criterion: {criterion} · max_depth={max_depth} · "
    f"min_samples_split={min_samples_split} · min_samples_leaf={min_samples_leaf} · "
    f"split={100-test_size}:{test_size}"
)

st.subheader("3. Confusion matrix")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
cm_fig, cm_ax = plt.subplots(figsize=(6, 4.5))
im = cm_ax.imshow(cm, interpolation="nearest")
cm_ax.set_title("Confusion Matrix — Decision Tree")
cm_ax.set_xlabel("Predicted label")
cm_ax.set_ylabel("Actual label")
cm_ax.set_xticks([0, 1], labels=TARGET_NAMES, rotation=10)
cm_ax.set_yticks([0, 1], labels=TARGET_NAMES)
for i in range(2):
    for j in range(2):
        cm_ax.text(j, i, str(cm[i, j]), ha="center", va="center")
cm_fig.colorbar(im, ax=cm_ax, fraction=0.046, pad=0.04)
cm_fig.tight_layout()
st.pyplot(cm_fig)
plt.close(cm_fig)

st.subheader("4. Classification report")
report = classification_report(
    y_test, y_pred, labels=[0, 1], target_names=TARGET_NAMES,
    output_dict=True, zero_division=0
)
st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"), use_container_width=True)

st.subheader("5. ROC curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_fig, roc_ax = plt.subplots(figsize=(7, 5))
roc_ax.plot(fpr, tpr, linewidth=2, label=f"Decision Tree (AUC = {auc_score:.3f})")
roc_ax.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
roc_ax.set_xlabel("False Positive Rate")
roc_ax.set_ylabel("True Positive Rate")
roc_ax.set_title("ROC Curve")
roc_ax.grid(True, alpha=0.25)
roc_ax.legend()
roc_fig.tight_layout()
st.pyplot(roc_fig)
plt.close(roc_fig)

st.subheader("6. Visualise the Decision Tree")
display_depth = st.slider("Depth shown in tree diagram", min_value=1, max_value=max(1, model.get_depth()), value=min(4, max(1, model.get_depth())))
tree_fig, tree_ax = plt.subplots(figsize=(22, 11))
plot_tree(
    model,
    feature_names=FEATURES,
    class_names=TARGET_NAMES,
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=display_depth,
    impurity=True,
    ax=tree_ax,
)
tree_ax.set_title(f"Decision Tree ({criterion}, max_depth={max_depth})")
tree_fig.tight_layout()
st.pyplot(tree_fig)
plt.close(tree_fig)

st.subheader("7. Human-readable decision rules")
rules = export_text(model, feature_names=FEATURES, max_depth=display_depth)
st.code(rules, language="text")

st.subheader("8. Feature importance")
importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=False)
importance_fig, importance_ax = plt.subplots(figsize=(8, 5))
importance_ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])
importance_ax.set_xlabel("Gini-based feature importance")
importance_ax.set_title("Feature Importance")
importance_ax.grid(axis="x", alpha=0.2)
importance_fig.tight_layout()
st.pyplot(importance_fig)
plt.close(importance_fig)
st.dataframe(importance_df, use_container_width=True, hide_index=True)

st.subheader("9. Effect of tree depth (overfitting and pruning)")
depth_rows = []
for depth in range(1, 16):
    candidate = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=int(random_state),
    )
    candidate.fit(X_train, y_train)
    depth_rows.append({
        "Max depth": depth,
        "Training accuracy": accuracy_score(y_train, candidate.predict(X_train)),
        "Testing accuracy": accuracy_score(y_test, candidate.predict(X_test)),
        "Actual tree depth": candidate.get_depth(),
    })
depth_df = pd.DataFrame(depth_rows)
depth_fig, depth_ax = plt.subplots(figsize=(8, 4.5))
depth_ax.plot(depth_df["Max depth"], depth_df["Training accuracy"], marker="o", label="Training accuracy")
depth_ax.plot(depth_df["Max depth"], depth_df["Testing accuracy"], marker="o", label="Testing accuracy")
depth_ax.set_xlabel("Maximum depth")
depth_ax.set_ylabel("Accuracy")
depth_ax.set_title("Train vs Test Accuracy by Tree Depth")
depth_ax.set_ylim(0, 1.05)
depth_ax.grid(True, alpha=0.25)
depth_ax.legend()
depth_fig.tight_layout()
st.pyplot(depth_fig)
plt.close(depth_fig)
best_row = depth_df.loc[depth_df["Testing accuracy"].idxmax()]
st.caption(f"Highest test accuracy in this depth sweep: {best_row['Testing accuracy']:.1%} at max_depth={int(best_row['Max depth'])}. This is a comparison on the same test set, not a separate validation set.")

with st.expander("View depth comparison table", expanded=False):
    st.dataframe(depth_df.style.format({"Training accuracy": "{:.3f}", "Testing accuracy": "{:.3f}"}), use_container_width=True, hide_index=True)

st.subheader("10. Predict using custom patient feature values")
st.warning(
    "Educational demonstration only. This model is not a medical diagnostic tool and must not be used "
    "to make healthcare decisions."
)
st.write("Enter values for all 13 features. Defaults are set to dataset medians.")
medians = df[FEATURES].median()
mins = df[FEATURES].min()
maxs = df[FEATURES].max()
with st.form("patient_prediction"):
    entered = {}
    left, right = st.columns(2)
    for i, feature in enumerate(FEATURES):
        container = left if i % 2 == 0 else right
        with container:
            entered[feature] = st.number_input(
                f"{feature} — {FEATURE_DESCRIPTIONS[feature]}",
                min_value=float(mins[feature]),
                max_value=float(maxs[feature]),
                value=float(medians[feature]),
                step=0.1 if df[feature].dtype.kind == "f" else 1.0,
            )
    submitted = st.form_submit_button("Predict class", type="primary")

if submitted:
    input_df = pd.DataFrame([entered], columns=FEATURES)
    predicted_class = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])
    if predicted_class == 1:
        st.success(f"Model output: **Disease class (1)** · estimated model probability: {probability:.1%}")
    else:
        st.info(f"Model output: **No Disease class (0)** · estimated model probability of disease: {probability:.1%}")
    st.caption("This output is for a machine-learning lab demonstration only; it is not a clinical assessment.")

with st.expander("View test-set predictions", expanded=False):
    pred_df = X_test.copy()
    pred_df["Actual target"] = y_test.map({0: "No Disease", 1: "Disease"})
    pred_df["Predicted target"] = pd.Series(y_pred, index=y_test.index).map({0: "No Disease", 1: "Disease"})
    pred_df["Predicted disease probability"] = y_prob
    pred_df["Correct"] = y_test.to_numpy() == y_pred
    st.dataframe(pred_df, use_container_width=True)
    st.download_button(
        "Download test predictions as CSV",
        data=pred_df.to_csv(index=True).encode("utf-8"),
        file_name="decision_tree_test_predictions.csv",
        mime="text/csv",
    )

st.subheader("11. Observations and conclusion")
st.markdown(
    f"""
**Observations**
- The dataset contains **{len(df)} complete records** and **13 input features**.
- The Decision Tree uses the **{criterion}** splitting criterion.
- With the current settings, test accuracy is **{accuracy:.1%}** and ROC-AUC is **{auc_score:.3f}**.
- The confusion matrix and classification report summarise correct and incorrect predictions.
- Feature importance and readable rules help explain the model's decisions.
- The depth comparison illustrates how a deeper tree can improve training accuracy while potentially overfitting.

**Conclusion**

A Decision Tree classifier was implemented using scikit-learn and the Heart Disease dataset.
The model was evaluated using accuracy, confusion matrix, classification report and ROC-AUC.
The tree plot, extracted rules and feature importance visualise the decision-making process.
Tree depth and minimum leaf/split settings demonstrate ways to control complexity and reduce overfitting.
"""
)
st.caption("Machine Learning Lab Manual by Dr. Sachin Balawant Takmare | Experiment No. 06")
