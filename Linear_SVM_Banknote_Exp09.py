import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.datasets import fetch_openml

st.set_page_config(page_title="Experiment 09 | Linear SVM", page_icon="📄", layout="wide")
st.title("Experiment No. 09 — Counterfeit Banknote Detection")
st.caption("Linear Support Vector Machine (SVM) · Flask/HTML-style lab workflow presented as a Streamlit UI")
st.info("Dataset: UCI Banknote Authentication. The target labels are 0 = genuine and 1 = counterfeit in this interface; verify the source dataset's label convention before formal reporting.")

@st.cache_data(show_spinner="Loading the Banknote Authentication dataset...")
def load_data():
    # UCI dataset is also mirrored on OpenML; internet is needed on first load.
    try:
        data = fetch_openml(data_id=1462, as_frame=True, parser="auto")
        df = data.frame.copy()
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
        df = pd.read_csv(url, header=None)
        df.columns = ["variance", "skewness", "curtosis", "entropy", "class"]
        return df

try:
    df = load_data()
except Exception as e:
    st.error("Could not download the dataset. Check your internet connection, or upload the UCI data file below.")
    uploaded = st.file_uploader("Upload data_banknote_authentication.txt or a CSV", type=["csv", "txt"])
    if uploaded is None:
        st.stop()
    df = pd.read_csv(uploaded, header=None)
    if df.shape[1] == 5:
        df.columns = ["variance", "skewness", "curtosis", "entropy", "class"]
    else:
        st.error("Expected four numerical feature columns and one class column.")
        st.stop()

# Normalize column naming for OpenML or UCI source
if "class" not in df.columns:
    target_candidates = [c for c in df.columns if str(c).lower() in ("class", "target", "y")]
    if target_candidates:
        df = df.rename(columns={target_candidates[0]: "class"})
    else:
        df.columns = ["variance", "skewness", "curtosis", "entropy", "class"]
features = [c for c in df.columns if c != "class"]
for c in features:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["class"] = pd.to_numeric(df["class"], errors="coerce")
df = df.dropna(subset=features + ["class"])
if len(features) != 4:
    st.warning("Dataset did not contain the expected four features. Using the first four numeric columns.")
    features = [c for c in df.select_dtypes(include=np.number).columns if c != "class"][:4]
X = df[features]
y = df["class"].astype(int)

with st.sidebar:
    st.header("Model settings")
    test_size = st.slider("Test set (%)", 10, 40, 20, 5) / 100
    c_value = st.select_slider("Regularization C", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
    show_data = st.checkbox("Show dataset preview", True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=c_value))
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
a,b,c = st.columns(3)
a.metric("Dataset samples", f"{len(df):,}")
b.metric("Test accuracy", f"{acc:.3%}")
c.metric("Support vectors", f"{int(model[-1].n_support_.sum()):,}")

if show_data:
    st.subheader("Dataset preview")
    st.dataframe(df.head(12), use_container_width=True)
st.subheader("Evaluation")
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
with right:
    st.code(classification_report(y_test, pred, zero_division=0), language="text")
st.subheader("Feature distributions")
selected = st.selectbox("Choose a feature to visualize", features)
fig, ax = plt.subplots()
for label in sorted(y.unique()):
    ax.hist(df.loc[y == label, selected], bins=30, alpha=0.55, label=f"Class {label}")
ax.set_xlabel(selected); ax.set_ylabel("Count"); ax.legend()
st.pyplot(fig)

st.subheader("Try a banknote sample")
st.caption("Enter numerical measurements from the dataset feature fields. Predictions are educational and are not a real-world authentication guarantee.")
cols = st.columns(4)
defaults = [float(X[c].median()) for c in features]
values = []
for i, feat in enumerate(features):
    with cols[i % 4]:
        values.append(st.number_input(str(feat), value=defaults[i], format="%.6f", key=f"sample_{feat}"))
if st.button("Classify banknote", type="primary"):
    result = int(model.predict(pd.DataFrame([values], columns=features))[0])
    st.success(f"Predicted class: {result} — " + ("Genuine (label 0)" if result == 0 else "Counterfeit (label 1)"))

st.caption("Method: train/test split → StandardScaler → SVC(kernel='linear') → accuracy, confusion matrix, and precision/recall/F1 report.")
