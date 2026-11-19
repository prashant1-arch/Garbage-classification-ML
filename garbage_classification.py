# GARBAGE CLASSIFICATION MODEL
# Logistic Regression - Biodegradable vs Non-Biodegradable

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load dataset
df = pd.read_csv("garbage_classification_dataset_5000.csv")

# 2. Understand the dataset
print("First five rows:")
print(df.head())

print("\nDataset shape (rows, columns):", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nClass distribution:")
print(df["Waste_Type"].value_counts())

# 3. Visualize the dataset
df["Waste_Type"].value_counts().plot(
    kind="bar", rot=0, title="Garbage Class Distribution"
)
plt.xlabel("Waste Type")
plt.ylabel("Number of records")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
for waste_type in df["Waste_Type"].unique():
    subset = df[df["Waste_Type"] == waste_type]
    plt.scatter(
        subset["Weight_g"],
        subset["Moisture_Percent"],
        label=waste_type,
        alpha=0.5
    )
plt.xlabel("Weight (g)")
plt.ylabel("Moisture (%)")
plt.title("Weight vs Moisture")
plt.legend()
plt.tight_layout()
plt.show()

# 4. Separate features (X) and target (y)
x = df[["Weight_g", "Moisture_Percent", "Size_cm3"]]
y = df["Waste_Type"]

print("\nFeatures (X):")
print(x.head())
print("\nTarget (y):")
print(y.head())

# 5. Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42, stratify=y
)

print("\nTraining records:", len(x_train))
print("Testing records:", len(x_test))

# 6. Create and train Logistic Regression model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(max_iter=1000))
])

model.fit(x_train, y_train)

# 7. Make predictions
y_pred = model.predict(x_test)

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
print("\nActual vs Predicted:")
print(comparison.head(20))

# 8. Evaluate the model
cm = confusion_matrix(
    y_test, y_pred,
    labels=["Biodegradable", "Non-Biodegradable"]
)
print("\nConfusion Matrix (rows = actual, columns = predicted):")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, pos_label="Biodegradable", zero_division=0
)
recall = recall_score(
    y_test, y_pred, pos_label="Biodegradable", zero_division=0
)
f1 = f1_score(
    y_test, y_pred, pos_label="Biodegradable", zero_division=0
)

print("\nAccuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1 Score:", round(f1, 3))

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 9. Predict the class of new garbage
new_garbage = pd.DataFrame({
    "Weight_g": [80],
    "Moisture_Percent": [65],
    "Size_cm3": [400]
})

predicted_class = model.predict(new_garbage)[0]
predicted_probabilities = model.predict_proba(new_garbage)[0]

print("\nNew garbage prediction:", predicted_class)
print("Class order:", model.classes_)
print("Class probabilities:", predicted_probabilities)
