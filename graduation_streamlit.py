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
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv('Crop_recommendation.csv')

# Encode labels
label_encoded = {crop: i for i, crop in enumerate(data['label'].unique())}
data['label'] = data['label'].map(label_encoded)

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

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

# Streamlit UI
st.title("Crop Recommendation System")

# Model selection
model_name = st.selectbox("Select a model", list(trained_models.keys()))
model = trained_models[model_name]

# Display model performance
st.subheader("Model Performance")
st.write(f"Train Accuracy: {model.score(X_train, y_train):.2f}")
st.write(f"Test Accuracy: {model.score(X_test, y_test):.2f}")

# Confusion matrix
st.subheader("Confusion Matrix")
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Classification report
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# Predict top 5 recommended crops
data_input = st.text_input("Enter soil data (comma separated values)")
if st.button("Recommend Crops"):
    try:
        user_data = np.array([list(map(float, data_input.split(',')))]).reshape(1, -1)
        user_data = scaler.transform(user_data)
        probs = model.predict_proba(user_data)
        top5_indices = np.argsort(probs, axis=1)[:, -5:][:, ::-1]
        plant_names = {i: name for name, i in label_encoded.items()}
        top5_plants = pd.DataFrame({
            "Rank": range(1, 6),
            "Plant Name": [plant_names[index] for index in top5_indices[0]],
            "Probability": [probs[0, index] for index in top5_indices[0]]
        })
        st.subheader("Top 5 Recommended Crops")
        st.table(top5_plants)
    except Exception as e:
        st.error(f"Error: {e}")
