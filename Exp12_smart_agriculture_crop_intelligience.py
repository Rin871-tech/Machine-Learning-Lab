import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score, ConfusionMatrixDisplay
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="AgriSense | Crop Intelligence", page_icon="🌱", layout="wide")

# Duolingo-inspired, light and friendly palette
st.markdown("""
<style>
:root { --green:#58CC02; --darkgreen:#2E7D32; --yellow:#FFC800; --blue:#1CB0F6; --ink:#24313B; --paper:#F7FBF3; }
.stApp { background: linear-gradient(180deg,#F7FBF3 0%,#FFFFFF 45%,#F4FAFF 100%); color:var(--ink); }
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1500px; }
h1,h2,h3 { color:#23412C; }
div[data-testid="stMetric"] { background:#FFFFFF; border:1px solid #DDEED4; padding:14px 16px; border-radius:16px; box-shadow:0 3px 10px #173a1b0a; }
div.stButton > button { background:#58CC02; color:#16330D; border:0; border-radius:12px; font-weight:700; }
div.stButton > button:hover { background:#49B800; color:#FFFFFF; border:0; }
[data-testid="stSidebar"] { background:#F0F9E9; }
div[data-testid="stAlert"] { border-radius:12px; }
.hero { background:linear-gradient(115deg,#E5F8D7,#E5F6FF); padding:26px 30px; border:1px solid #D3EBC8; border-radius:22px; margin-bottom:18px; }
.pill { display:inline-block; background:#E3F7D7; color:#27651C; border-radius:99px; padding:5px 10px; font-size:12px; font-weight:700; margin-right:5px; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🌱 AgriSense")
st.sidebar.caption("Smart Agriculture & Crop Intelligence")
pages = [
    "Overview", "01 · Data Explorer & EDA", "02–03 · Yield Regression",
    "04–08 · Crop Classification Lab", "Normalization & Standardization",
    "09–10 · SVM Kernel Lab", "11 · Decision Tree vs Random Forest",
    "Crop Recommendation Playground", "Dataset & Project Notes"
]
page = st.sidebar.radio("Navigate", pages)
st.sidebar.markdown("---")
st.sidebar.caption("Educational prototype · Uses public sample datasets and synthetic crop-like data.")

@st.cache_data
def make_agri_data(n=1800, seed=42):
    rng = np.random.default_rng(seed)
    crops = np.array(["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Pulses"])
    crop = rng.choice(crops, n, p=[.22,.20,.20,.13,.12,.13])
    rainfall = np.clip(rng.normal(780, 260, n), 150, 1800)
    temp = np.clip(rng.normal(27, 5.5, n), 10, 44)
    humidity = np.clip(rng.normal(64, 14, n), 20, 98)
    ph = np.clip(rng.normal(6.5, .8, n), 3.8, 9.2)
    nitrogen = np.clip(rng.normal(65, 24, n), 5, 150)
    phosphorus = np.clip(rng.normal(42, 18, n), 3, 110)
    potassium = np.clip(rng.normal(48, 21, n), 2, 130)
    area = np.clip(rng.normal(2.4, 1.1, n), .2, 8)
    crop_factor = pd.Series(crop).map({"Rice":1.2,"Wheat":1.0,"Maize":1.1,"Cotton":.8,"Sugarcane":1.35,"Pulses":.75}).to_numpy()
    suitability = (0.25*np.exp(-((temp-27)/10)**2) + .22*np.exp(-((ph-6.5)/1.4)**2) +
                   .22*np.exp(-((rainfall-850)/600)**2) + .16*(humidity/100) +
                   .15*np.minimum((nitrogen+phosphorus+potassium)/240,1))
    yield_t_ha = np.maximum(0.5, crop_factor*(1.0 + 4.2*suitability) + rng.normal(0,.35,n))
    risk_score = (np.abs(temp-27)/18 + np.abs(ph-6.5)/3 + np.maximum(0, 450-rainfall)/700 +
                  np.maximum(0, 25-nitrogen)/45 + rng.normal(0,.18,n))
    condition = np.where(risk_score < .8, "Favorable", np.where(risk_score < 1.45, "Moderate", "At Risk"))
    # occasional missing values to support data-cleaning demonstration
    df = pd.DataFrame({
        "crop":crop, "rainfall_mm":rainfall.round(1), "temperature_c":temp.round(1),
        "humidity_pct":humidity.round(1), "soil_ph":ph.round(2), "nitrogen":nitrogen.round(1),
        "phosphorus":phosphorus.round(1), "potassium":potassium.round(1),
        "farm_area_ha":area.round(2), "yield_t_ha":yield_t_ha.round(2),
        "crop_condition":condition
    })
    for col in ["rainfall_mm","soil_ph","nitrogen"]:
        df.loc[rng.choice(n, int(n*.015), replace=False), col] = np.nan
    return df

agri = make_agri_data()
NUM_COLS = ["rainfall_mm","temperature_c","humidity_pct","soil_ph","nitrogen","phosphorus","potassium","farm_area_ha"]
TARGET_REG = "yield_t_ha"
TARGET_CLF = "crop_condition"

def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def clean_frame(df):
    out = df.copy()
    for c in out.select_dtypes(include=np.number).columns:
        out[c] = out[c].fillna(out[c].median())
    for c in out.select_dtypes(exclude=np.number).columns:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].mode().iloc[0])
    return out

def metric_cards(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)

def classification_data(source="AgriSense crop conditions"):
    if source == "Iris flower benchmark":
        ds = load_iris(as_frame=True)
        X, y = ds.data, ds.target
        return X, y, list(ds.target_names), "Iris"
    if source == "Breast cancer benchmark":
        ds = load_breast_cancer(as_frame=True)
        return ds.data, ds.target, list(ds.target_names), "Breast cancer"
    df = clean_frame(agri)
    X = df[NUM_COLS]
    y = LabelEncoder().fit_transform(df[TARGET_CLF])
    return X, pd.Series(y, name=TARGET_CLF), ["At Risk", "Favorable", "Moderate"], "AgriSense crop conditions"

def make_clf_model(name, seed, scale=True):
    estimator = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=160, max_depth=None, random_state=seed, class_weight="balanced"),
        "Linear SVM": SVC(kernel="linear", probability=True, C=1.0, random_state=seed),
        "Polynomial SVM": SVC(kernel="poly", degree=3, probability=True, C=1.0, random_state=seed),
        "RBF SVM": SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=seed),
    }[name]
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale and name in ["Logistic Regression","K-Nearest Neighbors","Linear SVM","Polynomial SVM","RBF SVM"]:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)

def fit_classifier(X, y, model_name, seed=42, test_size=.25):
    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)
    model = make_clf_model(model_name, seed)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return model, Xtr, Xte, ytr, yte, pred

def fit_tree_rf_classification(X, y, seed, test_size, tree_depth, n_estimators):
    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)
    models = {
        "Decision Tree": Pipeline([("imputer",SimpleImputer(strategy="median")),("model",DecisionTreeClassifier(max_depth=tree_depth, random_state=seed,class_weight="balanced"))]),
        "Random Forest": Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=n_estimators,max_depth=tree_depth or None,random_state=seed,class_weight="balanced"))])
    }
    results = []
    fitted = {}
    for name, model in models.items():
        model.fit(Xtr,ytr); pred=model.predict(Xte); fitted[name]=model
        results.append({"Model":name,"Accuracy":accuracy_score(yte,pred),"Precision (weighted)":precision_score(yte,pred,average="weighted",zero_division=0),"Recall (weighted)":recall_score(yte,pred,average="weighted",zero_division=0),"F1 (weighted)":f1_score(yte,pred,average="weighted",zero_division=0)})
    return Xtr,Xte,ytr,yte,fitted,pd.DataFrame(results)

def fit_tree_rf_regression(X, y, seed, test_size, tree_depth, n_estimators):
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=seed)
    models = {
        "Decision Tree": Pipeline([("imputer",SimpleImputer(strategy="median")),("model",DecisionTreeRegressor(max_depth=tree_depth,random_state=seed))]),
        "Random Forest": Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestRegressor(n_estimators=n_estimators,max_depth=tree_depth or None,random_state=seed,n_jobs=-1))])
    }
    results=[]; fitted={}; preds={}
    for name,model in models.items():
        model.fit(Xtr,ytr); pred=model.predict(Xte); fitted[name]=model; preds[name]=pred
        results.append({"Model":name,"MAE":mean_absolute_error(yte,pred),"MSE":mean_squared_error(yte,pred),"RMSE":mean_squared_error(yte,pred)**.5,"R²":r2_score(yte,pred)})
    return Xtr,Xte,ytr,yte,fitted,preds,pd.DataFrame(results)

if page == "Overview":
    st.markdown("""
    <div class="hero">
      <span class="pill">11 EXPERIMENTS IN ONE APP</span><span class="pill">CLASSIFICATION + REGRESSION</span>
      <h1>🌱 AgriSense: Smart Agriculture & Crop Intelligence</h1>
      <p>Explore agricultural data, estimate crop yield, assess crop-condition risk, and compare machine-learning models in one friendly dashboard.</p>
    </div>
    """, unsafe_allow_html=True)
    metric_cards([("Sample records", f"{len(agri):,}"),("Input features",len(NUM_COLS)),("Crop categories",agri["crop"].nunique()),("ML experiment modules","11")])
    st.markdown("### Your workflow")
    cols=st.columns(4)
    for col, title, desc in zip(cols,["Explore","Prepare","Train","Compare"],["Understand trends and missing data","Impute, encode and scale features","Fit regression and classification models","Review metrics, plots and feature importance"]):
        with col:
            st.markdown(f"#### {title}")
            st.write(desc)
    st.info("The built-in AgriSense dataset is **synthetic** and intended for demonstrating ML workflows. It is not an agronomic source and must not be used to make real farming decisions.")
    st.markdown("### Modules mapped to the lab")
    st.dataframe(pd.DataFrame([
        ["01","Pandas, Matplotlib, Seaborn, EDA","Data Explorer & EDA"],
        ["02","Simple Linear Regression","Yield Regression"],
        ["03","Multiple Linear Regression","Yield Regression"],
        ["04","Binary Logistic Regression","Crop Classification Lab"],
        ["05","Multiclass Logistic Regression","Crop Classification Lab"],
        ["06","Decision Tree","Crop Classification Lab + Exp 11"],
        ["07","K-Nearest Neighbors","Crop Classification Lab"],
        ["08","Gaussian Naive Bayes","Crop Classification Lab"],
        ["Preprocessing","Normalization and Standardization","Scaling Lab"],
        ["09","Linear SVM","SVM Kernel Lab"],
        ["10","Polynomial and RBF SVM","SVM Kernel Lab"],
        ["11","Decision Tree vs Random Forest: classification and regression","Dedicated comparison module"],
    ],columns=["Experiment","Concept","Dashboard location"]),hide_index=True,use_container_width=True)

elif page == "01 · Data Explorer & EDA":
    st.title("01 · Data Explorer & Exploratory Data Analysis")
    st.write("Inspect data quality, distributions, correlations and crop-group differences. Upload your own CSV if desired.")
    uploaded=st.file_uploader("Optional: upload a CSV dataset",type=["csv"])
    df=agri.copy() if uploaded is None else pd.read_csv(uploaded)
    st.caption(f"Active dataset: {'AgriSense sample' if uploaded is None else uploaded.name}")
    c1,c2,c3=st.columns(3)
    c1.metric("Rows",f"{len(df):,}"); c2.metric("Columns",len(df.columns)); c3.metric("Missing cells",int(df.isna().sum().sum()))
    with st.expander("Preview dataset",expanded=True):
        st.dataframe(df.head(30),use_container_width=True)
    a,b=st.columns(2)
    with a:
        st.subheader("Missing values by column")
        miss=df.isna().sum().sort_values(ascending=False)
        fig,ax=plt.subplots(figsize=(8,4)); miss[miss>0].plot(kind="bar",ax=ax); ax.set_ylabel("Missing cells"); ax.set_xlabel(""); show_fig(fig)
    with b:
        st.subheader("Numeric feature distributions")
        numeric=df.select_dtypes(include=np.number).columns.tolist()
        if numeric:
            feature=st.selectbox("Choose a numeric feature",numeric)
            fig,ax=plt.subplots(figsize=(8,4)); sns.histplot(df[feature].dropna(),kde=True,ax=ax); ax.set_title(feature); show_fig(fig)
    st.subheader("Descriptive statistics")
    st.dataframe(df.describe(include="all").transpose(),use_container_width=True)
    numeric=df.select_dtypes(include=np.number)
    if len(numeric.columns)>1:
        st.subheader("Correlation heatmap")
        fig,ax=plt.subplots(figsize=(10,6)); sns.heatmap(numeric.corr(),annot=True,fmt=".2f",cmap="YlGnBu",ax=ax); show_fig(fig)
    cat=[c for c in df.columns if df[c].dtype=="object"]
    if cat and numeric.shape[1]:
        cat_col=st.selectbox("Group numeric values by",cat)
        num_col=st.selectbox("Numeric value to compare",numeric.columns.tolist(),key="eda_num")
        fig,ax=plt.subplots(figsize=(9,4)); sns.boxplot(data=df,x=cat_col,y=num_col,ax=ax); ax.tick_params(axis="x",rotation=25); show_fig(fig)
    st.download_button("Download active dataset as CSV",df.to_csv(index=False).encode(),file_name="agrisense_dataset.csv",mime="text/csv")

elif page == "02–03 · Yield Regression":
    st.title("02–03 · Crop Yield Regression")
    st.write("Compare simple linear regression (one input) with multiple linear regression (several inputs).")
    df=clean_frame(agri)
    simple_feature=st.selectbox("Simple regression feature",["rainfall_mm","temperature_c","soil_ph","nitrogen","farm_area_ha"])
    multi_features=st.multiselect("Multiple regression features",NUM_COLS,default=["rainfall_mm","temperature_c","soil_ph","nitrogen","phosphorus","potassium"])
    if not multi_features: multi_features=[simple_feature]
    test_size=st.slider("Test set size",0.15,0.40,0.25,0.05)
    seed=st.number_input("Random seed",min_value=0,max_value=9999,value=42)
    results=[]
    fitted={}
    for name, features in [("Simple Linear Regression",[simple_feature]),("Multiple Linear Regression",multi_features)]:
        X=df[features]; y=df[TARGET_REG]
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=int(seed))
        model=Pipeline([("imputer",SimpleImputer(strategy="median")),("model",LinearRegression())])
        model.fit(Xtr,ytr); pred=model.predict(Xte); fitted[name]=(model,Xte,yte,pred,features)
        results.append({"Model":name,"Features":len(features),"MAE":mean_absolute_error(yte,pred),"MSE":mean_squared_error(yte,pred),"RMSE":mean_squared_error(yte,pred)**.5,"R²":r2_score(yte,pred)})
    st.dataframe(pd.DataFrame(results).round(4),hide_index=True,use_container_width=True)
    chosen=st.selectbox("Visualize model",list(fitted.keys()))
    model,Xte,yte,pred,features=fitted[chosen]
    fig,ax=plt.subplots(figsize=(8,4)); ax.scatter(yte,pred,alpha=.65); lo=min(float(yte.min()),float(pred.min())); hi=max(float(yte.max()),float(pred.max())); ax.plot([lo,hi],[lo,hi],"--"); ax.set_xlabel("Actual yield (synthetic t/ha)"); ax.set_ylabel("Predicted yield"); ax.set_title(chosen+" · Actual vs predicted"); show_fig(fig)
    st.write("Model features:",", ".join(features))
    st.caption("R² can be negative on a test split. These values evaluate this synthetic demonstration dataset only.")

elif page == "04–08 · Crop Classification Lab":
    st.title("04–08 · Crop Condition Classification")
    st.write("Try binary or multiclass classification with Logistic Regression, Decision Tree, KNN and Naive Bayes.")
    source=st.selectbox("Dataset",["AgriSense crop conditions","Iris flower benchmark","Breast cancer benchmark"])
    X,y,names,display_name=classification_data(source)
    if source=="Breast cancer benchmark":
        st.warning("Breast cancer is a standard educational benchmark, unrelated to agriculture. Use AgriSense for the crop-themed demonstration.")
    model_name=st.selectbox("Algorithm",["Logistic Regression","Decision Tree","K-Nearest Neighbors","Gaussian Naive Bayes"])
    test_size=st.slider("Test split",.15,.40,.25,.05,key="clf_split")
    seed=st.number_input("Random seed",min_value=0,max_value=9999,value=42,key="clf_seed")
    model,Xtr,Xte,ytr,yte,pred=fit_classifier(X,y,model_name,int(seed),test_size)
    metric_cards([("Accuracy",f"{accuracy_score(yte,pred):.3f}"),("Precision (weighted)",f"{precision_score(yte,pred,average='weighted',zero_division=0):.3f}"),("Recall (weighted)",f"{recall_score(yte,pred,average='weighted',zero_division=0):.3f}"),("F1 (weighted)",f"{f1_score(yte,pred,average='weighted',zero_division=0):.3f}")])
    a,b=st.columns(2)
    with a:
        st.subheader("Confusion matrix")
        fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay.from_predictions(yte,pred,ax=ax,colorbar=False); show_fig(fig)
    with b:
        st.subheader("Classification report")
        st.code(classification_report(yte,pred,zero_division=0))
    st.subheader("Feature importance (when supported)")
    estimator=model.named_steps["model"]
    if hasattr(estimator,"feature_importances_"):
        imp=pd.Series(estimator.feature_importances_,index=X.columns).sort_values()
        fig,ax=plt.subplots(figsize=(8,5)); imp.tail(15).plot(kind="barh",ax=ax); ax.set_xlabel("Importance"); show_fig(fig)
    else:
        st.caption("This algorithm does not expose tree-style feature importance. Use the Experiment 11 module for tree-based importance.")

elif page == "Normalization & Standardization":
    st.title("Data Scaling · Normalization vs Standardization")
    st.write("Compare the original numeric features with Min-Max scaling and z-score standardization.")
    df=clean_frame(agri)
    cols=st.multiselect("Features",NUM_COLS,default=["rainfall_mm","temperature_c","soil_ph","nitrogen"])
    if cols:
        original=df[cols]
        normalized=pd.DataFrame(MinMaxScaler().fit_transform(original),columns=cols)
        standardized=pd.DataFrame(StandardScaler().fit_transform(original),columns=cols)
        tab1,tab2,tab3=st.tabs(["Original","Min-Max normalized","Standardized"])
        for tab,data in [(tab1,original),(tab2,normalized),(tab3,standardized)]:
            with tab:
                st.dataframe(data.describe().round(3),use_container_width=True)
                fig,ax=plt.subplots(figsize=(9,4)); sns.boxplot(data=data,ax=ax); ax.tick_params(axis="x",rotation=20); show_fig(fig)
        st.info("Min-Max scaling maps each selected feature to [0, 1]. Standardization transforms features to approximately mean 0 and standard deviation 1. Fit scalers on training data only in production pipelines to avoid data leakage.")

elif page == "09–10 · SVM Kernel Lab":
    st.title("09–10 · Support Vector Machine Kernel Lab")
    st.write("Compare linear, polynomial and RBF kernels. Synthetic data keeps the kernel shapes easy to explore.")
    dataset=st.selectbox("Dataset",["Synthetic moons","Synthetic circles","AgriSense crop conditions"])
    if dataset=="Synthetic moons":
        X,y=make_moons(n_samples=900,noise=.23,random_state=42); X=pd.DataFrame(X,columns=["feature_1","feature_2"])
    elif dataset=="Synthetic circles":
        X,y=make_circles(n_samples=900,noise=.12,factor=.45,random_state=42); X=pd.DataFrame(X,columns=["feature_1","feature_2"])
    else:
        X,y,_,_=classification_data()
    kernel=st.selectbox("Kernel",["linear","poly","rbf"])
    c_value=st.slider("C (regularization)",.1,10.0,1.0,.1)
    gamma=st.selectbox("Gamma",["scale","auto",0.1,1.0,5.0])
    test_size=st.slider("Test split",.15,.40,.25,.05,key="svm_split")
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=42,stratify=y)
    svm=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",SVC(kernel=kernel,C=c_value,gamma=gamma,degree=3))])
    svm.fit(Xtr,ytr); pred=svm.predict(Xte)
    metric_cards([("Accuracy",f"{accuracy_score(yte,pred):.3f}"),("Precision",f"{precision_score(yte,pred,average='weighted',zero_division=0):.3f}"),("Recall",f"{recall_score(yte,pred,average='weighted',zero_division=0):.3f}"),("F1",f"{f1_score(yte,pred,average='weighted',zero_division=0):.3f}")])
    fig,ax=plt.subplots(figsize=(7,5))
    if X.shape[1]==2:
        ax.scatter(X.iloc[:,0],X.iloc[:,1],c=y,cmap="viridis",s=18,alpha=.7); ax.set_xlabel(X.columns[0]); ax.set_ylabel(X.columns[1]); ax.set_title(f"{dataset} · SVM {kernel} kernel")
    else:
        ax.text(.05,.7,"More than 2 features selected.\nBoundary visualization omitted.",transform=ax.transAxes)
    show_fig(fig)
    st.subheader("Classification report")
    st.code(classification_report(yte,pred,zero_division=0))
    st.caption("Experiment 09 originally mentions the UCI Banknote Authentication dataset. This combined app demonstrates linear SVM on synthetic and crop-condition data; add the downloaded banknote CSV to reproduce that exact dataset-specific lab.")

elif page == "11 · Decision Tree vs Random Forest":
    st.title("11 · Decision Tree vs Random Forest")
    st.write("Fair comparison: both models use the same train/test split, input features and target. Tune the tree depth and forest size.")
    task=st.radio("Task",["Classification","Regression"],horizontal=True)
    test_size=st.slider("Test set size",.15,.40,.25,.05,key="exp11_test")
    seed=st.number_input("Random seed",min_value=0,max_value=9999,value=42,key="exp11_seed")
    depth_option=st.selectbox("Maximum depth",["Unlimited",3,5,8,12])
    depth=None if depth_option=="Unlimited" else int(depth_option)
    n_estimators=st.slider("Random Forest trees",50,300,160,10)
    if task=="Classification":
        source=st.selectbox("Classification dataset",["AgriSense crop conditions","Iris flower benchmark"],key="exp11_clf_source")
        X,y,labels,_=classification_data(source)
        Xtr,Xte,ytr,yte,models,results=fit_tree_rf_classification(X,y,int(seed),test_size,depth,n_estimators)
        st.dataframe(results.round(4),hide_index=True,use_container_width=True)
        st.bar_chart(results.set_index("Model")[["Accuracy","Precision (weighted)","Recall (weighted)","F1 (weighted)"]])
        chosen=st.selectbox("Inspect model",list(models.keys()),key="exp11_clf_model")
        pred=models[chosen].predict(Xte)
        a,b=st.columns(2)
        with a:
            st.subheader("Confusion matrix")
            fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay.from_predictions(yte,pred,ax=ax,colorbar=False); show_fig(fig)
        with b:
            st.subheader("Classification report")
            st.code(classification_report(yte,pred,zero_division=0))
        est=models[chosen].named_steps["model"]
        if hasattr(est,"feature_importances_"):
            imp=pd.Series(est.feature_importances_,index=X.columns).sort_values()
            fig,ax=plt.subplots(figsize=(8,5)); imp.tail(15).plot(kind="barh",ax=ax); ax.set_title(chosen+" feature importance"); show_fig(fig)
        results_csv=results.to_csv(index=False).encode()
        st.download_button("Download comparison metrics",results_csv,"exp11_classification_comparison.csv","text/csv")
    else:
        df=clean_frame(agri)
        target=st.selectbox("Regression target",["yield_t_ha","farm_area_ha"])
        features=st.multiselect("Input features",NUM_COLS,default=["rainfall_mm","temperature_c","humidity_pct","soil_ph","nitrogen","phosphorus","potassium"])
        if not features: features=["rainfall_mm","temperature_c","soil_ph"]
        X=df[features]; y=df[target]
        Xtr,Xte,ytr,yte,models,preds,results=fit_tree_rf_regression(X,y,int(seed),test_size,depth,n_estimators)
        st.dataframe(results.round(4),hide_index=True,use_container_width=True)
        st.bar_chart(results.set_index("Model")[["MAE","RMSE","R²"]])
        chosen=st.selectbox("Inspect model",list(models.keys()),key="exp11_reg_model")
        pred=preds[chosen]
        fig,ax=plt.subplots(figsize=(8,4)); ax.scatter(yte,pred,alpha=.7); lo=min(float(yte.min()),float(pred.min())); hi=max(float(yte.max()),float(pred.max())); ax.plot([lo,hi],[lo,hi],"--"); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title(chosen+" · Actual vs predicted"); show_fig(fig)
        est=models[chosen].named_steps["model"]
        if hasattr(est,"feature_importances_"):
            imp=pd.Series(est.feature_importances_,index=X.columns).sort_values()
            fig,ax=plt.subplots(figsize=(8,5)); imp.plot(kind="barh",ax=ax); ax.set_title(chosen+" feature importance"); show_fig(fig)
        out=pd.DataFrame({"actual":yte.to_numpy(),"decision_tree_prediction":preds["Decision Tree"],"random_forest_prediction":preds["Random Forest"]})
        st.download_button("Download predictions",out.to_csv(index=False).encode(),"exp11_regression_predictions.csv","text/csv")
    st.markdown("#### How to interpret")
    st.write("- **Decision Tree** is easy to visualize but can overfit if too deep.")
    st.write("- **Random Forest** averages many trees and often generalizes better, at the cost of interpretability and extra computation.")
    st.write("- There is no guaranteed winner: compare held-out metrics on your selected dataset.")

elif page == "Crop Recommendation Playground":
    st.title("Crop Intelligence Playground")
    st.write("Explore how environmental inputs relate to the synthetic crop-condition class and yield estimate.")
    a,b,c=st.columns(3)
    with a:
        rainfall=st.slider("Rainfall (mm)",150,1800,800,25)
        temp=st.slider("Temperature (°C)",10,44,27)
        humidity=st.slider("Humidity (%)",20,98,64)
    with b:
        ph=st.slider("Soil pH",3.8,9.2,6.5,.1)
        nitrogen=st.slider("Nitrogen",5,150,65)
        phosphorus=st.slider("Phosphorus",3,110,42)
    with c:
        potassium=st.slider("Potassium",2,130,48)
        area=st.slider("Farm area (ha)",.2,8.0,2.4,.1)
        crop=st.selectbox("Crop type",["Rice","Wheat","Maize","Cotton","Sugarcane","Pulses"])
    df=clean_frame(agri)
    X=df[NUM_COLS]; y=df[TARGET_CLF]
    model=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced"))])
    model.fit(X,y)
    sample=pd.DataFrame([[rainfall,temp,humidity,ph,nitrogen,phosphorus,potassium,area]],columns=NUM_COLS)
    pred=model.predict(sample)[0]
    probs=model.predict_proba(sample)[0]
    metric_cards([("Estimated condition",str(pred)),("Highest class probability",f"{max(probs):.1%}"),("Selected crop",crop)])
    st.subheader("Model class probabilities")
    prob_df=pd.DataFrame({"Condition":model.classes_,"Probability":probs}).sort_values("Probability",ascending=False).set_index("Condition")
    st.bar_chart(prob_df)
    st.warning("Illustrative ML output only. The training labels are generated from synthetic rules and are not a validated crop diagnosis or recommendation.")
    st.download_button("Download this input as CSV",sample.to_csv(index=False).encode(),"crop_playground_input.csv","text/csv")

else:
    st.title("Dataset & Project Notes")
    st.markdown("""
    ### Dataset strategy
    - **AgriSense sample data:** generated deterministically in the app so the project runs without downloading files.
    - **Optional CSV upload:** use the EDA page to inspect your own data.
    - **Benchmark datasets:** Iris and Breast Cancer from scikit-learn are included for demonstrating standard classification algorithms.
    - **Exact UCI Banknote Authentication requirement:** the combined app does not bundle this external dataset. Download it separately if your instructor requires that specific dataset, then adapt the SVM module or use it as a standalone practical.

    ### Data and model caveats
    - The crop data is synthetic, not collected from farms.
    - Metrics are for learning and comparison, not production agronomy.
    - Do not use test data for fitting imputers or scalers in a real project. Use pipelines and cross-validation for robust evaluation.
    - Real crop recommendation should incorporate local soil tests, crop calendars, geography, irrigation, season, and validated domain expertise.

    ### Suggested future improvements
    1. Add verified district-level weather and soil datasets with source citations.
    2. Add model persistence and a model-card page.
    3. Add cross-validation and hyperparameter search.
    4. Add a proper crop recommendation dataset and evaluate generalization.
    5. Deploy the dashboard after checking data privacy and model limitations.
    """)
    st.markdown("### Feature dictionary")
    st.dataframe(pd.DataFrame([
        ["rainfall_mm","Rainfall (mm)","Input"],
        ["temperature_c","Temperature (°C)","Input"],
        ["humidity_pct","Relative humidity (%)","Input"],
        ["soil_ph","Soil acidity/alkalinity indicator","Input"],
        ["nitrogen","Synthetic soil nitrogen value","Input"],
        ["phosphorus","Synthetic soil phosphorus value","Input"],
        ["potassium","Synthetic soil potassium value","Input"],
        ["farm_area_ha","Farm area (hectares)","Input"],
        ["yield_t_ha","Synthetic yield (tonnes/hectare)","Regression target"],
        ["crop_condition","Synthetic crop condition category","Classification target"],
    ],columns=["Column","Meaning","Role"]),hide_index=True,use_container_width=True)
