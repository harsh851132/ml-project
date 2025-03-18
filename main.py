import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.datasets import fetch_openml

st.set_page_config(page_title="ML Comparative Study", layout="wide")
st.title("Comparative Study of Machine Learning Models")

with st.status("Loading models and preparing data..."):
    time.sleep(2)  # Simulate loading time

    diabetes = fetch_openml(name="diabetes", version=1, as_frame=True)
    df = diabetes.data
    df['target'] = diabetes.target.astype('category').cat.codes

    selected_features = ["mass", "age", "plas", "pres", "pedi"]
    X = df[selected_features]
    y = df['target']

    st.sidebar.header("User Input Parameters")
    test_size = st.sidebar.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": (LogisticRegression(), {'C': [0.1, 1, 10], 'max_iter': [100, 200, 500]}),
        "Random Forest": (RandomForestClassifier(), {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}),
        "SVM": (SVC(probability=True), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),  
    }

    # Train Models & Evaluate
    results = []
    conf_matrices = {}
    trained_models = {}

    for name, (model, params) in models.items():
        clf = GridSearchCV(model, params, scoring='f1', cv=5)
        clf.fit(X_train, y_train)
        y_pred = clf.best_estimator_.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({"Model": name, "Accuracy": accuracy, "F1 Score": f1, "Best Params": clf.best_params_})
        conf_matrices[name] = confusion_matrix(y_test, y_pred)
        trained_models[name] = clf.best_estimator_

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

st.success("Models loaded successfully!")

st.subheader("View Dataset")
st.dataframe(df.head())

csv = df.to_csv(index=False)
st.download_button(label="Download Dataset as CSV", data=csv, file_name="diabetes_dataset.csv", mime="text/csv")

st.subheader("Model Performance Summary")
st.dataframe(results_df)

st.subheader("Model Accuracy Comparison")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Model", y="Accuracy", data=results_df, palette="viridis", ax=ax)
ax.set_ylim(0, 1)
st.pyplot(fig)

st.subheader("Confusion Matrices")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (model_name, cm) in zip(axes, conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Make Predictions")
input_data = {}
for feature in selected_features:
    input_data[feature] = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))

if st.button("Predict Diabetes Status"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    
    st.write("### Predictions from Different Models:")
    for model_name, model in trained_models.items():
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] if hasattr(model, 'predict_proba') else "N/A"
        result_text = "Diabetes Detected" if prediction == 1 else "No Diabetes"
        st.write(f"**{model_name}:** {result_text} (Probability: {probability:.2f} if applicable)")
