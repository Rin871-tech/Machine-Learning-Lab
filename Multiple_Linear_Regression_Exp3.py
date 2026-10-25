# -*- coding: utf-8 -*-
"""
Experiment: Implementation of Multiple Linear Regression
Dataset: Real Estate Valuation (reconstructed from the professor-provided PDF)
Based on the professor-provided Real_estate_price_prediction.ipynb.

Install:
    python3 -m pip install -r requirements_multiple_regression.txt

Run:
    streamlit run Multiple_Linear_Regression_Exp3.py

Keep Real_estate_extracted.csv in the same folder, or upload it in the sidebar.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Multiple Linear Regression | ML Lab",
    page_icon="🏠",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main-title {font-size: 2rem; font-weight: 750; margin-bottom: .2rem;}
      .subtitle {color: #64748b; margin-bottom: 1rem;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22);
        padding: .8rem; border-radius: .7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🏠 Implementation of Multiple Linear Regression</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Real estate price prediction using more than two independent features</div>',
    unsafe_allow_html=True,
)

with st.expander("Aim, objectives and theory", expanded=False):
    st.markdown(
        r"""
**Aim:** To implement multiple linear regression using Python and evaluate its performance.

**Objectives**
1. Study multiple linear regression.
2. Train a model using more than two independent features.
3. Evaluate performance using Mean Squared Error (MSE) and R².
4. Visualise actual versus predicted values and residuals.
5. Predict house price per unit area from user-entered property features.

**Theory**

Multiple linear regression estimates a continuous target from two or more independent variables:

\[
\hat{y} = b_0 + b_1x_1 + b_2x_2 + \cdots + b_px_p
\]

Here, \(b_0\) is the intercept, \(b_i\) are coefficients, and \(x_i\) are input features.
MSE measures the average squared prediction error; lower is better. R² measures how much
of the target variation is explained by the model; values closer to 1 generally indicate a
better fit on the evaluated data.
"""
    )

with st.expander("Procedure followed (based on the professor's notebook)", expanded=False):
    st.markdown(
        """
1. Import pandas, NumPy, Matplotlib, and scikit-learn.
2. Load the real-estate dataset.
3. Use **house age**, **distance to the nearest station**, **number of convenience stores**,
   **latitude**, and **longitude** as independent variables.
4. Use **house price of unit area** as the dependent variable.
5. Split the data into training and testing sets (80:20 by default, random state 42).
6. Train `sklearn.linear_model.LinearRegression`.
7. Predict values for the test set.
8. Calculate the coefficients, intercept, MSE, and R².
9. Visualise model performance and use the fitted model for an interactive prediction.
"""
    )


# Load extracted data from the attached PDF, unless the user uploads a CSV.
def load_local_csv():
    candidates = [
        Path(__file__).resolve().parent / "Real_estate_extracted.csv",
        Path.cwd() / "Real_estate_extracted.csv",
        Path.cwd() / "Real estate.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return pd.read_csv(candidate), candidate.name
            except Exception:
                pass
    return None, None


st.sidebar.header("⚙️ Dataset & model settings")
uploaded = st.sidebar.file_uploader("Upload real-estate CSV (optional)", type=["csv"])
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
            "Place `Real_estate_extracted.csv` beside this Python file, or upload a CSV "
            "in the sidebar. The PDF is not read directly at runtime."
        )
        st.stop()

st.sidebar.caption(f"Dataset: {source_name}")

# Standardize possible column names without changing the professor's chosen features.
df = raw_df.copy()
df.columns = [str(c).strip() for c in df.columns]

expected_features = [
    "house age",
    "distance to the nearest station",
    "number of convenience stores",
    "latitude",
    "longitude",
]
target_name = "house price of unit area"

# Remove a possible unnamed index column, but preserve all meaningful columns.
unnamed = [c for c in df.columns if c.lower().startswith("unnamed:")]
if unnamed:
    df = df.drop(columns=unnamed)

# Make numeric-looking columns numeric where possible.
for col in df.columns:
    if df[col].dtype == object:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            df[col] = converted

if target_name not in df.columns:
    st.error(
        f"Target column `{target_name}` was not found. Check that the uploaded CSV has "
        "the same column names as the professor's real-estate dataset."
    )
    st.write("Columns found:", list(df.columns))
    st.stop()

missing_features = [c for c in expected_features if c not in df.columns]
if missing_features:
    st.error("Missing expected feature columns: " + ", ".join(missing_features))
    st.write("Columns found:", list(df.columns))
    st.stop()

# Professor's notebook uses exactly these five predictors and this target.
features = expected_features
available_df = df[features + [target_name]].copy()
for col in features + [target_name]:
    available_df[col] = pd.to_numeric(available_df[col], errors="coerce")
available_df = available_df.replace([np.inf, -np.inf], np.nan).dropna()

if len(available_df) < 10:
    st.error("At least 10 complete numeric rows are required to train and evaluate the model.")
    st.stop()

st.sidebar.subheader("Training configuration")
test_percent = st.sidebar.slider("Test set (%)", min_value=10, max_value=40, value=20, step=5)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

X = available_df[features]
y = available_df[target_name]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_percent / 100, random_state=int(random_state)
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = float(np.sqrt(mse))
r2 = r2_score(y_test, y_pred)

st.subheader("1. Dataset overview")
a, b, c, d = st.columns(4)
a.metric("Rows in source", f"{len(raw_df):,}")
b.metric("Complete rows used", f"{len(available_df):,}")
c.metric("Training samples", f"{len(X_train):,}")
d.metric("Testing samples", f"{len(X_test):,}")

with st.expander("View dataset and descriptive statistics", expanded=False):
    st.dataframe(available_df.head(20), use_container_width=True)
    st.write("Descriptive statistics")
    st.dataframe(available_df.describe().T, use_container_width=True)

st.subheader("2. Model evaluation")
m1, m2, m3 = st.columns(3)
m1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
m2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
m3.metric("R² Score", f"{r2:.4f}")

st.caption(
    f"Evaluation uses the held-out test set ({len(X_test)} rows). "
    f"Split: {100-test_percent}% train / {test_percent}% test; random state: {int(random_state)}."
)

st.subheader("3. Model coefficients")
coefficients_df = pd.DataFrame({
    "Independent feature": features,
    "Coefficient": model.coef_,
})
st.dataframe(
    coefficients_df.style.format({"Coefficient": "{:.6f}"}),
    use_container_width=True,
    hide_index=True,
)
st.metric("Intercept", f"{model.intercept_:.6f}")
st.markdown("**Fitted equation**")
equation_parts = [f"({coef:.4f} × {name})" for name, coef in zip(features, model.coef_)]
equation = f"Predicted price = {model.intercept_:.4f} + " + " + ".join(equation_parts)
st.code(equation, language="text")

st.subheader("4. Actual vs predicted prices")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.7)
low = float(min(np.min(y_test), np.min(y_pred)))
high = float(max(np.max(y_test), np.max(y_pred)))
ax.plot([low, high], [low, high], linestyle="--", linewidth=2, label="Ideal prediction")
ax.set_xlabel("Actual house price of unit area")
ax.set_ylabel("Predicted house price of unit area")
ax.set_title("Actual vs Predicted Values")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("5. Residual analysis")
residuals = np.asarray(y_test) - np.asarray(y_pred)
res_fig, res_ax = plt.subplots(figsize=(8, 4.5))
res_ax.scatter(y_pred, residuals, alpha=0.7)
res_ax.axhline(0, linestyle="--", linewidth=2)
res_ax.set_xlabel("Predicted price")
res_ax.set_ylabel("Residual (actual − predicted)")
res_ax.set_title("Residuals vs Predicted Values")
res_ax.grid(True, alpha=0.25)
res_fig.tight_layout()
st.pyplot(res_fig)
plt.close(res_fig)

with st.expander("View test-set predictions", expanded=False):
    predictions_df = pd.DataFrame({
        "Actual price": np.asarray(y_test),
        "Predicted price": y_pred,
        "Residual": residuals,
    }, index=y_test.index).sort_index()
    st.dataframe(predictions_df.style.format("{:.4f}"), use_container_width=True)
    csv_bytes = predictions_df.to_csv(index=True).encode("utf-8")
    st.download_button(
        "Download test predictions as CSV",
        data=csv_bytes,
        file_name="multiple_regression_test_predictions.csv",
        mime="text/csv",
    )

st.subheader("6. Predict house price")
st.write("Enter property details below. The values start at the dataset medians.")
median_values = available_df[features].median()
min_values = available_df[features].min()
max_values = available_df[features].max()

with st.form("prediction_form"):
    input_values = {}
    col1, col2 = st.columns(2)
    for i, feature in enumerate(features):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            step = 1.0 if feature == "number of convenience stores" else 0.1
            input_values[feature] = st.number_input(
                feature.title(),
                min_value=float(min_values[feature]),
                max_value=float(max_values[feature]),
                value=float(median_values[feature]),
                step=step,
                format="%.5f" if feature in ("latitude", "longitude") else "%.3f",
                help=f"Observed data range: {min_values[feature]:.5f} to {max_values[feature]:.5f}",
            )
    submitted = st.form_submit_button("Predict house price", type="primary")

if submitted:
    input_df = pd.DataFrame([input_values], columns=features)
    predicted_price = float(model.predict(input_df)[0])
    st.success(f"Estimated house price of unit area: **{predicted_price:.2f}**")
    st.caption(
        "This is a model estimate based on the five selected features, not a guaranteed market price."
    )

st.subheader("7. Observations and conclusion")
st.markdown(
    f"""
**Observations**
- The model was trained using **{len(X_train)}** samples and tested using **{len(X_test)}** samples.
- Five independent features were used: house age, distance to the nearest station,
  number of convenience stores, latitude, and longitude.
- The test-set MSE is **{mse:.4f}**, RMSE is **{rmse:.4f}**, and R² is **{r2:.4f}**.
- The coefficient table shows the learned contribution associated with each feature
  while holding the other included features constant.

**Conclusion**

Multiple linear regression was implemented using scikit-learn and the real-estate dataset.
The model predicts house price of unit area from more than two independent features.
MSE and R² were used to evaluate test-set performance, and the visualisations help compare
actual and predicted values and inspect residual errors.
"""
)

st.caption("Machine Learning Lab | Multiple Linear Regression | Based on the professor-provided notebook")
