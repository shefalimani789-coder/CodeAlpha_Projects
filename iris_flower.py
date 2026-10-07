# ============================================================
# TASK 1: IRIS FLOWER CLASSIFICATION
# ============================================================

# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# STEP 1: Load the Iris Dataset
# ------------------------------------------------------------

iris = load_iris()

# Input features
X = iris.data

# Target/output
y = iris.target

# Flower species names
target_names = iris.target_names

print("Iris Flower Classification")
print("=" * 50)

print("\nFeatures:")
print(iris.feature_names)

print("\nFlower Classes:")
print(target_names)


# ------------------------------------------------------------
# STEP 2: Split Dataset into Training and Testing Data
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# STEP 3: Scale the Features
# ------------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ------------------------------------------------------------
# STEP 4: Create and Train the Machine Learning Model
# ------------------------------------------------------------

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# ------------------------------------------------------------
# STEP 5: Make Predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# STEP 6: Calculate Accuracy
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


# ------------------------------------------------------------
# STEP 7: Classification Report
# ------------------------------------------------------------

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=target_names
))


# ------------------------------------------------------------
# STEP 8: Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# ------------------------------------------------------------
# STEP 9: Display Confusion Matrix Graph
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.title("Iris Flower Classification - Confusion Matrix")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 10: Test the Model with a New Flower
# ------------------------------------------------------------

# Measurements:
# [sepal length, sepal width, petal length, petal width]

new_flower = [[5.1, 3.5, 1.4, 0.2]]

# Scale the new flower using the same scaler
new_flower_scaled = scaler.transform(new_flower)

# Predict the species
prediction = model.predict(new_flower_scaled)

print("\nNew Flower Prediction:")
print("Measurements:", new_flower[0])
print("Predicted Species:", target_names[prediction[0]])

print("\nProgram completed successfully!")