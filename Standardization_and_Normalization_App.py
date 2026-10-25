# -*- coding: utf-8 -*-
"""
Standardization and Normalization — Interactive ML Lab App
Based on the professor-provided Standardization_and_Normalization.ipynb.

The app covers:
1. Standardization with StandardScaler on CGPA and IQ from the placement dataset.
2. Normalization with MinMaxScaler on Alcohol and Malic acid from the wine dataset.

Install:
    python3 -m pip install -r requirements_scaling.txt

Run:
    streamlit run Standardization_and_Normalization_App.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

st.set_page_config(
    page_title="Standardization & Normalization | ML Lab",
    page_icon="📏",
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
    '<div class="main-title">📏 Standardization and Normalization</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Feature scaling using StandardScaler and MinMaxScaler</div>',
    unsafe_allow_html=True,
)

with st.expander("Aim, objectives and theory", expanded=False):
    st.markdown(
        r"""
**Aim:** To understand and implement standardization and normalization using Python.

**Standardization — StandardScaler**

Standardization transforms a feature to have a mean close to 0 and standard deviation close to 1:

\[
z = \frac{x-\mu}{\sigma}
\]

Here, \(x\) is an observed value, \(\mu\) is the mean learned from the training data, and
\(\sigma\) is the standard deviation learned from the training data.

**Normalization — MinMaxScaler**

Min-max normalization rescales a feature to a specified range, usually [0, 1]:

\[
x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}
\]

The minimum and maximum values are learned from the training data. Scaling parameters must be
fit on training data only and then applied to test data to avoid data leakage.
"""
    )

with st.expander("Procedure followed (based on the professor's notebook)", expanded=False):
    st.markdown(
        """
**Part A — Standardization**
1. Load the Placement_Y_N dataset.
2. Select CGPA and IQ as features.
3. Split the data into training and testing sets using `test_size=0.2` and `random_state=42`.
4. Fit `StandardScaler` on the training features only.
5. Transform both training and test sets.
6. Compare descriptive statistics, scatter plots, and feature distributions before and after scaling.

**Part B — Normalization**
1. Load the wine dataset and select Class label, Alcohol, and Malic acid.
2. Use Alcohol and Malic acid as features and Class label for plotting.
3. Split the features and target using `test_size=0.3` and `random_state=0`.
4. Fit `MinMaxScaler` on training features only.
5. Transform training and test sets.
6. Compare statistics, scatter plots, and distributions before and after normalization.
"""
    )


def load_csv_candidates(filename):
    candidates = [
        Path(__file__).resolve().parent / filename,
        Path.cwd() / filename,
    ]
    for path in candidates:
        if path.exists():
            try:
                return pd.read_csv(path), path.name
            except Exception:
                continue
    return None, None


def kde_or_hist(ax, values, label, color=None):
    values = pd.Series(values).dropna()
    if len(values) >= 3 and values.nunique() > 1:
        try:
            values.plot(kind="kde", ax=ax, label=label, color=color)
            return
        except Exception:
            pass
    ax.hist(values, bins=12, density=True, alpha=0.4, label=label, color=color)


def scaling_summary(original, scaled, feature_names):
    rows = []
    for name in feature_names:
        rows.append({
            "Feature": name,
            "Original mean": float(np.mean(original[name])),
            "Original std": float(np.std(original[name], ddof=1)),
            "Scaled mean": float(np.mean(scaled[name])),
            "Scaled std": float(np.std(scaled[name], ddof=1)),
            "Original min": float(np.min(original[name])),
            "Original max": float(np.max(original[name])),
            "Scaled min": float(np.min(scaled[name])),
            "Scaled max": float(np.max(scaled[name])),
        })
    return pd.DataFrame(rows)


st.sidebar.header("⚙️ Dataset and display settings")
part = st.sidebar.radio(
    "Choose experiment part",
    ["Standardization — Placement dataset", "Normalization — Wine dataset"],
)
if part.startswith("Standardization"):
    dataset_filename = "Placement_Y_N_scaling.csv"
    uploaded = st.sidebar.file_uploader("Upload placement CSV (optional)", type=["csv"], key="placement_upload")
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            source = uploaded.name
        except Exception as exc:
            st.error(f"Could not read placement CSV: {exc}")
            st.stop()
    else:
        raw, source = load_csv_candidates(dataset_filename)
        if raw is None:
            st.warning("Keep `Placement_Y_N_scaling.csv` beside this Python file or upload a placement CSV.")
            st.stop()

    raw.columns = [str(c).strip().lower() for c in raw.columns]
    # Accept the source file if it has an extra unnamed index column.
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("unnamed:")], errors="ignore")
    required = ["cgpa", "iq", "placement"]
    if not all(c in raw.columns for c in required):
        st.error("Placement data must contain columns named `cgpa`, `iq`, and `placement`.")
        st.write("Columns found:", list(raw.columns))
        st.stop()
    for c in required:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    data = raw[required].replace([np.inf, -np.inf], np.nan).dropna()
    feature_names = ["cgpa", "iq"]
    X = data[feature_names].copy()
    target = data["placement"].astype(int)
    test_size, random_state = 0.2, 42
    scaler = StandardScaler()
    scaling_name = "StandardScaler"
    target_names = {0: "Not Placed", 1: "Placed"}
    palette = None
    part_title = "Part A — Standardization using StandardScaler"
    part_description = "CGPA and IQ are standardized using parameters learned from the training set."
else:
    uploaded = st.sidebar.file_uploader("Upload wine CSV (optional)", type=["csv"], key="wine_upload")
    if uploaded is not None:
        try:
            raw_wine = pd.read_csv(uploaded)
            source = uploaded.name
        except Exception as exc:
            st.error(f"Could not read wine CSV: {exc}")
            st.stop()
    else:
        raw_wine, source = load_csv_candidates("wine_data_scaling.csv")
        if raw_wine is None:
            st.warning("Keep `wine_data_scaling.csv` beside this Python file or upload a wine CSV.")
            st.stop()

    # Support the provided full wine CSV and the 3-column version used in the notebook.
    lower_to_actual = {str(c).strip().lower().replace("_", " ").replace("-", " "): c for c in raw_wine.columns}
    class_col = lower_to_actual.get("class label") or lower_to_actual.get("class_label")
    alcohol_col = lower_to_actual.get("alcohol")
    malic_col = lower_to_actual.get("malic acid") or lower_to_actual.get("malic_acid")
    if class_col is not None and alcohol_col is not None and malic_col is not None:
        raw_wine = raw_wine[[class_col, alcohol_col, malic_col]].copy()
        raw_wine.columns = ["Class label", "Alcohol", "Malic acid"]
    elif raw_wine.shape[1] >= 3:
        # Notebook expects a headerless CSV in the first three columns.
        raw_wine = raw_wine.iloc[:, :3].copy()
        raw_wine.columns = ["Class label", "Alcohol", "Malic acid"]
    else:
        st.error("Wine data must provide class label, alcohol, and malic acid columns.")
        st.stop()
    for c in ["Class label", "Alcohol", "Malic acid"]:
        raw_wine[c] = pd.to_numeric(raw_wine[c], errors="coerce")
    data = raw_wine.replace([np.inf, -np.inf], np.nan).dropna()
    feature_names = ["Alcohol", "Malic acid"]
    X = data[feature_names].copy()
    target = data["Class label"].astype(int)
    test_size, random_state = 0.3, 0
    scaler = MinMaxScaler()
    scaling_name = "MinMaxScaler"
    target_names = {int(k): f"Class {int(k)}" for k in sorted(target.unique())}
    palette = None
    part_title = "Part B — Normalization using MinMaxScaler"
    part_description = "Alcohol and Malic acid are scaled to the [0, 1] range using training-set minimums and maximums."

st.sidebar.caption(f"Dataset loaded: {source}")
st.sidebar.caption(f"Rows after cleaning: {len(data)}")

if len(data) < 10:
    st.error("At least 10 valid rows are required.")
    st.stop()

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=test_size, random_state=random_state
    )
except ValueError as exc:
    st.error(f"Could not split the data: {exc}")
    st.stop()

# Fit scaler only on training data, then transform train and test.
X_train_scaled_np = scaler.fit_transform(X_train)
X_test_scaled_np = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=feature_names, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled_np, columns=feature_names, index=X_test.index)

st.header(part_title)
st.write(part_description)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Dataset rows", f"{len(data):,}")
m2.metric("Features scaled", len(feature_names))
m3.metric("Training rows", f"{len(X_train):,}")
m4.metric("Testing rows", f"{len(X_test):,}")

st.subheader("1. Original dataset")
with st.expander("View data and descriptive statistics", expanded=False):
    st.dataframe(data.head(20), use_container_width=True)
    st.write("Original feature statistics")
    st.dataframe(X.describe().T, use_container_width=True)
    st.write("Training feature statistics")
    st.dataframe(X_train.describe().T, use_container_width=True)

st.subheader("2. Scaling summary")
summary = scaling_summary(X_train, X_train_scaled, feature_names)
st.dataframe(
    summary.style.format({
        "Original mean": "{:.4f}", "Original std": "{:.4f}",
        "Scaled mean": "{:.4f}", "Scaled std": "{:.4f}",
        "Original min": "{:.4f}", "Original max": "{:.4f}",
        "Scaled min": "{:.4f}", "Scaled max": "{:.4f}",
    }),
    use_container_width=True,
    hide_index=True,
)
if scaling_name == "StandardScaler":
    st.info("After standardization, each training feature should have mean ≈ 0 and standard deviation ≈ 1.")
else:
    st.info("After MinMaxScaler, each training feature should have minimum 0 and maximum 1. Test values can fall outside [0, 1] if they exceed the training range.")

st.subheader("3. Compare feature distributions before and after scaling")
selected_feature = st.selectbox("Select feature", feature_names)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
kde_or_hist(axes[0], X_train[selected_feature], "Original")
axes[0].set_title(f"{selected_feature} — Before scaling")
axes[0].set_xlabel("Original value")
axes[0].grid(True, alpha=0.2)
kde_or_hist(axes[1], X_train_scaled[selected_feature], "Scaled")
axes[1].set_title(f"{selected_feature} — After {scaling_name}")
axes[1].set_xlabel("Scaled value")
axes[1].grid(True, alpha=0.2)
axes[0].legend()
axes[1].legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("4. Compare scatter plots")
x_feature = st.selectbox("X-axis feature", feature_names, index=0)
y_feature = st.selectbox("Y-axis feature", feature_names, index=1)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
if part.startswith("Normalization"):
    axes[0].scatter(X_train[x_feature], X_train[y_feature], c=y_train, alpha=0.8)
    axes[1].scatter(X_train_scaled[x_feature], X_train_scaled[y_feature], c=y_train, alpha=0.8)
    axes[0].set_title("Before Normalization")
    axes[1].set_title("After MinMax Normalization")
else:
    axes[0].scatter(X_train[x_feature], X_train[y_feature], c=y_train, alpha=0.8)
    axes[1].scatter(X_train_scaled[x_feature], X_train_scaled[y_feature], c=y_train, alpha=0.8)
    axes[0].set_title("Before Standardization")
    axes[1].set_title("After Standardization")
for ax, xlab, ylab in [
    (axes[0], x_feature, y_feature),
    (axes[1], f"{x_feature} (scaled)", f"{y_feature} (scaled)")
]:
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.2)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("5. Training and test data after scaling")
tab1, tab2 = st.tabs(["Scaled training data", "Scaled test data"])
with tab1:
    st.dataframe(X_train_scaled.head(20), use_container_width=True)
    st.download_button(
        "Download scaled training data",
        data=X_train_scaled.assign(target=y_train).to_csv(index=False).encode("utf-8"),
        file_name="scaled_training_data.csv",
        mime="text/csv",
    )
with tab2:
    st.dataframe(X_test_scaled.head(20), use_container_width=True)
    st.download_button(
        "Download scaled test data",
        data=X_test_scaled.assign(target=y_test).to_csv(index=False).encode("utf-8"),
        file_name="scaled_test_data.csv",
        mime="text/csv",
    )

with st.expander("Scaler parameters learned from training data", expanded=False):
    if scaling_name == "StandardScaler":
        st.write("Training means (`mean_`)")
        st.json({f: float(v) for f, v in zip(feature_names, scaler.mean_)})
        st.write("Training standard deviations (`scale_`)")
        st.json({f: float(v) for f, v in zip(feature_names, scaler.scale_)})
    else:
        st.write("Training minimums (`data_min_`)")
        st.json({f: float(v) for f, v in zip(feature_names, scaler.data_min_)})
        st.write("Training maximums (`data_max_`)")
        st.json({f: float(v) for f, v in zip(feature_names, scaler.data_max_)})
        st.write("Feature ranges (`data_range_`)")
        st.json({f: float(v) for f, v in zip(feature_names, scaler.data_range_)})

st.subheader("6. Observations and conclusion")
if scaling_name == "StandardScaler":
    conclusion_text = (
        "Standardization was performed using StandardScaler. The scaler learned means and standard deviations "
        "from the training subset and applied the same transformation to training and test features. "
        "The resulting training features have mean close to zero and standard deviation close to one."
    )
    obs_1 = "CGPA and IQ originally use different scales."
    obs_2 = "StandardScaler shifts and scales each feature based on the training-set mean and standard deviation."
else:
    conclusion_text = (
        "Normalization was performed using MinMaxScaler. The scaler learned feature minimums and maximums "
        "from the training subset and applied the same transformation to training and test features. "
        "Training feature values are mapped to the [0, 1] range."
    )
    obs_1 = "Alcohol and Malic acid originally use different numeric ranges."
    obs_2 = "MinMaxScaler maps each training feature's minimum to 0 and maximum to 1."
st.markdown(
    f"""
**Observations**
- {obs_1}
- {obs_2}
- Scaling parameters were fitted on the training data only to prevent data leakage.
- Scatter plots and distribution plots show the effect of scaling on feature values.

**Conclusion**

{conclusion_text}
"""
)

st.caption("Machine Learning Lab | Standardization and Normalization | Based on the professor-provided notebook")
