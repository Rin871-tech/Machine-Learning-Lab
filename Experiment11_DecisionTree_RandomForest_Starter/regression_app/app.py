
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="California House Value Predictor",
    page_icon="🏡",
    layout="wide",
)

st.title("🏡 California House Value Predictor")
st.caption(
    "Experiment No. 11 | Decision Tree Regressor vs Random Forest Regressor"
)

st.info(
    "Educational demonstration using historical California housing data "
    "from the 1990 U.S. Census. This is not a current property valuation tool."
)


# =========================================================
# 2. LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    dataset = fetch_california_housing(as_frame=True)

    df = dataset.frame.copy()

    # Target values are in units of $100,000.
    # Convert to dollars only when displaying predictions.
    features = list(dataset.feature_names)
    target = "MedHouseVal"

    df = df.drop_duplicates().copy()
    df = df.dropna(subset=features + [target]).copy()

    return df, features, target


try:
    df, features, target = load_data()
except Exception as error:
    st.error(
        "Unable to load the California Housing dataset. "
        "The first run requires internet access."
    )
    st.exception(error)
    st.stop()


X = df[features]
y = df[target]


# =========================================================
# 3. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


# =========================================================
# 4. TRAIN MODELS
# =========================================================

@st.cache_resource
def train_models(_X_train, _y_train):

    decision_tree = DecisionTreeRegressor(
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
    )

    random_forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    decision_tree.fit(_X_train, _y_train)
    random_forest.fit(_X_train, _y_train)

    return {
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
    }


models = train_models(X_train, y_train)


# =========================================================
# 5. EVALUATION METRICS
# =========================================================

def evaluate_model(model):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R² Score": r2,
    }


metrics = pd.DataFrame(
    {
        name: evaluate_model(model)
        for name, model in models.items()
    }
).T


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("Experiment 11")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview & EDA",
        "Model Comparison",
        "Prediction Analysis",
        "Feature Importance",
        "Predict House Value",
    ],
)

st.sidebar.markdown("---")
st.sidebar.write("**Dataset:** California Housing")
st.sidebar.write(f"**Records:** {len(df):,}")
st.sidebar.write(f"**Features:** {len(features)}")
st.sidebar.write("**Target:** Median house value")
st.sidebar.write("**Test split:** 20%")
st.sidebar.write("**Random seed:** 42")


# =========================================================
# 7. OVERVIEW AND EDA
# =========================================================

if page == "Overview & EDA":

    st.header("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total records", f"{len(df):,}")
    c2.metric("Input features", len(features))
    c3.metric(
        "Missing values",
        int(df[features + [target]].isna().sum().sum()),
    )
    c4.metric(
        "Mean target value",
        f"${y.mean() * 100000:,.0f}",
    )

    st.subheader("Dataset preview")
    st.dataframe(df.head(15), use_container_width=True)

    st.subheader("Statistical summary")
    st.dataframe(
        df.describe().T,
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("House value distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.hist(
            y * 100000,
            bins=40,
            edgecolor="black",
        )

        ax.set_xlabel("Median house value ($)")
        ax.set_ylabel("Number of districts")
        ax.set_title("Distribution of target values")

        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Median income vs house value")

        sample = df.sample(
            n=min(2500, len(df)),
            random_state=42,
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.scatterplot(
            data=sample,
            x="MedInc",
            y="MedHouseVal",
            alpha=0.45,
            s=18,
            ax=ax,
        )

        ax.set_xlabel("Median income (tens of thousands of USD)")
        ax.set_ylabel("Median house value (units of $100,000)")
        ax.set_title("Income and housing value")

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Feature correlation")

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        df[features + [target]].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
    )

    ax.set_title("Correlation between numerical features")
    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# 8. MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.header("Decision Tree vs Random Forest")

    st.write(
        "Both regressors use the same training and test sets. "
        "MAE and RMSE measure prediction error, while R² measures "
        "how well the model explains variation in the target."
    )

    st.subheader("Evaluation results")

    st.dataframe(
        metrics.style.format("{:.4f}"),
        use_container_width=True,
    )

    st.caption(
        "Lower MAE and RMSE are better. Higher R² is generally better."
    )

    st.subheader("MAE and RMSE comparison")

    fig, ax = plt.subplots(figsize=(9, 5))

    metrics[["MAE", "RMSE"]].plot(
        kind="bar",
        ax=ax,
        rot=0,
    )

    ax.set_ylabel("Error (units of $100,000)")
    ax.set_xlabel("Algorithm")
    ax.set_title("Regression error comparison")
    ax.grid(axis="y", alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("R² comparison")

    fig, ax = plt.subplots(figsize=(7, 4))

    metrics["R² Score"].plot(
        kind="bar",
        ax=ax,
        rot=0,
    )

    ax.set_ylabel("R² Score")
    ax.set_xlabel("Algorithm")
    ax.set_title("Coefficient of determination")
    ax.axhline(0, linewidth=1)
    ax.grid(axis="y", alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)

    best_mae = metrics["MAE"].idxmin()
    best_rmse = metrics["RMSE"].idxmin()
    best_r2 = metrics["R² Score"].idxmax()

    c1, c2, c3 = st.columns(3)

    c1.metric("Lowest MAE", best_mae)
    c2.metric("Lowest RMSE", best_rmse)
    c3.metric("Highest R²", best_r2)

    st.markdown("### Interpretation")

    st.write(
        "**Decision Tree Regressor:** Predicts values using a single "
        "tree of feature-based decisions. It is relatively easy to interpret."
    )

    st.write(
        "**Random Forest Regressor:** Averages predictions from multiple "
        "trees, which can reduce variance and improve generalization."
    )


# =========================================================
# 9. PREDICTION ANALYSIS
# =========================================================

elif page == "Prediction Analysis":

    st.header("Actual vs Predicted Values")

    selected_model = st.selectbox(
        "Choose a model",
        list(models.keys()),
    )

    model = models[selected_model]
    predictions = model.predict(X_test)

    actual_dollars = y_test.to_numpy() * 100000
    predicted_dollars = predictions * 100000

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        actual_dollars,
        predicted_dollars,
        alpha=0.45,
        s=18,
    )

    lower = min(actual_dollars.min(), predicted_dollars.min())
    upper = max(actual_dollars.max(), predicted_dollars.max())

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
    )

    ax.set_xlabel("Actual median house value ($)")
    ax.set_ylabel("Predicted median house value ($)")
    ax.set_title(f"{selected_model}: Actual vs Predicted")
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    residuals = actual_dollars - predicted_dollars

    st.subheader("Residual distribution")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        residuals,
        bins=40,
        edgecolor="black",
    )

    ax.axvline(0, linestyle="--")
    ax.set_xlabel("Residual: actual − predicted ($)")
    ax.set_ylabel("Number of districts")
    ax.set_title("Prediction error distribution")

    st.pyplot(fig)
    plt.close(fig)

    st.write(
        "Points closer to the diagonal represent more accurate estimates. "
        "Residuals near zero indicate smaller prediction errors."
    )


# =========================================================
# 10. FEATURE IMPORTANCE
# =========================================================

elif page == "Feature Importance":

    st.header("Feature Importance")

    selected_model = st.selectbox(
        "Choose a model",
        list(models.keys()),
    )

    model = models[selected_model]

    st.write(
        "Permutation importance measures how much model performance "
        "changes when one feature is shuffled on the test dataset."
    )

    with st.spinner("Calculating feature importance..."):

        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )

    importance = pd.Series(
        result.importances_mean,
        index=features,
    ).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))

    importance.plot(
        kind="barh",
        ax=ax,
    )

    ax.set_xlabel(
        "Increase in MAE when feature is shuffled "
        "(target units of $100,000)"
    )

    ax.set_ylabel("Feature")
    ax.set_title(f"{selected_model} — Permutation Importance")
    ax.axvline(0, linewidth=1)
    ax.grid(axis="x", alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Larger positive values generally indicate more useful features. "
        "Importance does not prove causation."
    )


# =========================================================
# 11. INTERACTIVE HOUSE VALUE PREDICTION
# =========================================================

elif page == "Predict House Value":

    st.header("Estimate Median House Value")

    st.write(
        "Enter the characteristics of a California census district. "
        "Both models will estimate its historical median house value."
    )

    descriptions = {
        "MedInc": "Median income (in tens of thousands of USD)",
        "HouseAge": "Median house age (years)",
        "AveRooms": "Average rooms per household",
        "AveBedrms": "Average bedrooms per household",
        "Population": "District population",
        "AveOccup": "Average household occupancy",
        "Latitude": "Latitude",
        "Longitude": "Longitude",
    }

    defaults = {
        "MedInc": 3.5,
        "HouseAge": 28.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 36.0,
        "Longitude": -119.0,
    }

    user_input = {}

    with st.form("housing_prediction_form"):

        cols = st.columns(2)

        for index, feature in enumerate(features):

            with cols[index % 2]:

                minimum = float(X[feature].min())
                maximum = float(X[feature].max())

                default_value = float(
                    np.clip(
                        defaults[feature],
                        minimum,
                        maximum,
                    )
                )

                user_input[feature] = st.number_input(
                    descriptions[feature],
                    min_value=minimum,
                    max_value=maximum,
                    value=default_value,
                    format="%.3f",
                    help=f"Dataset feature: {feature}",
                )

        submitted = st.form_submit_button(
            "Estimate house value",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        input_df = pd.DataFrame(
            [user_input],
            columns=features,
        )

        st.subheader("Estimated median house values")

        columns = st.columns(2)

        for col, (name, model) in zip(columns, models.items()):

            predicted_value = float(
                model.predict(input_df)[0]
            )

            # California Housing target is in $100,000 units.
            predicted_dollars = predicted_value * 100000

            with col:

                st.markdown(f"### {name}")

                st.metric(
                    "Estimated median value",
                    f"${predicted_dollars:,.0f}",
                )

                st.caption(
                    f"Raw prediction: {predicted_value:.4f} "
                    "units of $100,000"
                )

        st.warning(
            "The target is a district-level median value from historical "
            "data, not the asking price or present-day value of an individual "
            "property. The dataset's target is capped at 5.0 "
            "(approximately $500,000), which limits predictions for "
            "higher-value districts."
        )


# =========================================================
# 12. FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Experiment No. 11 | California Housing Dataset | "
    "Decision Tree Regressor vs Random Forest Regressor"
)
