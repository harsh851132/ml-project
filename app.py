import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Train-test split
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "MLP Classifier": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)
}

# Train models and collect accuracy
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

# Streamlit App
st.title("Comparative Study of ML Models")
st.write("This app compares the accuracy of different ML models on the same dataset.")

# Show dataset
if st.checkbox("Show Dataset"):
    st.write(df)

# Show model performance
st.subheader("Model Accuracy Comparison")
for name, acc in results.items():
    st.write(f"{name}: {acc:.4f}")

# Visualize results
st.subheader("Accuracy Bar Chart")
fig, ax = plt.subplots()
sns.barplot(x=list(results.keys()), y=list(results.values()), ax=ax)
ax.set_ylabel("Accuracy")
st.pyplot(fig)

# Model selection for prediction
st.subheader("Make a Prediction")
selected_model = st.selectbox("Choose a model", list(models.keys()))
input_features = []
for feature in data.feature_names:
    input_features.append(st.number_input(f"{feature}", value=0.0))

if st.button("Predict"):
    model = models[selected_model]
    input_array = np.array(input_features).reshape(1, -1)
    input_array = scaler.transform(input_array)
    prediction = model.predict(input_array)
    st.write(f"Predicted Class: {data.target_names[prediction[0]]}")
