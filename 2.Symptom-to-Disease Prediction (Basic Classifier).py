# Install Required Libraries********
# pip install pandas numpy scikit-learn

# Load Dataset************
import pandas as pd

# Load dataset
df = pd.read_csv("symptom_disease_dataset.csv")

print(df.head())

 # Data Preprocessing*********
from sklearn.preprocessing import LabelEncoder

# Separate features and target
X = df.drop("disease", axis=1)
y = df["disease"]

# Encode target labels (disease names → numbers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Classes:", le.classes_)

# Train-Test Split*************
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train Models
# 📌 Logistic Regression
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

# Random Forest******
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

# Evaluation**********
from sklearn.metrics import accuracy_score, classification_report

# Logistic Regression Predictions
lr_preds = lr_model.predict(X_test)

# Random Forest Predictions
rf_preds = rf_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_preds))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))

print("\nRandom Forest Report:\n", classification_report(y_test, rf_preds))

# Predict New Patient Case********
import numpy as np

# New input (reshape required)
new_patient = np.array([[1, 0, 1, 1]])

# Predict using Random Forest
prediction = rf_model.predict(new_patient)

# Convert back to disease name
predicted_disease = le.inverse_transform(prediction)

print("Predicted Disease:", predicted_disease[0])

# Feature Importance (Random Forest)**********
import matplotlib.pyplot as plt

importance = rf_model.feature_importances_

plt.barh(X.columns, importance)
plt.xlabel("Importance")
plt.title("Symptom Importance")
plt.show()

