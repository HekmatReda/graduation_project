import subprocess
import sys

# Ensure required libraries are installed
required_libraries = ["seaborn", "pandas", "numpy", "scikit-learn", "matplotlib", "streamlit"]
for lib in required_libraries:
    subprocess.call([sys.executable, "-m", "pip", "install", lib])

# Import libraries after installation
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Crop_recommendation.csv")

data = load_data()

st.title("Crop Recommendation Model Comparison")

# Display dataset preview
st.write("### Dataset Preview")
st.dataframe(data.head())

# Encode labels
label_encoded = {
    "rice": 0, "maize": 1, "jute": 2, "cotton": 3, "coconut": 4, "papaya": 5, "orange": 6, "apple": 7,
    "muskmelon": 8, "watermelon": 9, "grapes": 10, "mango": 11, "banana": 12, "pomegranate": 13, "lentil": 14,
    "blackgram": 15, "mungbean": 16, "mothbeans": 17, "pigeonpeas": 18, "kidneybeans": 19, "chickpea": 20, "coffee": 21
}
data['label'] = data['label'].map(label_encoded).astype(int)

# Split dataset
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=15, criterion='gini'),
    "SVM": SVC(kernel='linear', max_iter=10000, C=1, probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm='kd_tree'),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=12, random_state=42)
}

results = {}

st.write("### Model Training and Evaluation")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    results[name] = {"Train Score": train_score, "Test Score": test_score}
    
    st.write(f"**{name} Results:**")
    st.write(f"Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    st.pyplot(fig)
    
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

# Accuracy comparison
st.write("### Model Accuracy Comparison")

df_results = pd.DataFrame.from_dict(results, orient="index").reset_index().rename(columns={"index": "Model"})
df_melted = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Model", y="Score", hue="Metric", data=df_melted, palette="Set2", ax=ax)
plt.title("Model Accuracy Comparison (Train vs Test)")
plt.xlabel("Model")
plt.ylabel("Accuracy")
st.pyplot(fig)

# Predict top 5 crops for a random test sample
st.write("### Top 5 Recommended Crops")

random_sample = X_test[np.random.randint(0, X_test.shape[0])].reshape(1, -1)

selected_model = st.selectbox("Select Model for Prediction", list(models.keys()))
predicted_probs = models[selected_model].predict_proba(random_sample)

top5_indices = np.argsort(predicted_probs, axis=1)[:, -5:][:, ::-1][0]

plant_names = {i: name for name, i in label_encoded.items()}
top5_plants = pd.DataFrame({
    "Rank": range(1, 6),
    "Plant Name": [plant_names[i] for i in top5_indices],
    "Probability": [predicted_probs[0, i] for i in top5_indices]
})

st.write("Top 5 recommended crops based on soil data:")
st.dataframe(top5_plants)
