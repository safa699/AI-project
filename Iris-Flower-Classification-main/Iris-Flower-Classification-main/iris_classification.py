# ============================================
# IRIS FLOWER CLASSIFICATION USING MACHINE LEARNING
# Models: Logistic Regression, Naive Bayes,
#         Random Forest Classifier
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================
# Load Dataset
# ============================================

iris = load_iris()

X = iris.data
y = iris.target

# Create DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)

# Convert numeric labels into flower names
df["species"] = pd.Categorical.from_codes(y, iris.target_names)

print("\nFirst Five Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ============================================
# Data Visualization
# ============================================

sns.pairplot(df, hue="species")
plt.savefig("images/pairplot.png")
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(df.drop("species", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("images/heatmap.png")
plt.show()

# ============================================
# Train Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# Logistic Regression
# ============================================

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# ============================================
# Gaussian Naive Bayes
# ============================================

nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

# ============================================
# Random Forest Classifier
# ============================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ============================================
# Accuracy Comparison
# ============================================

lr_acc = accuracy_score(y_test, lr_pred)
nb_acc = accuracy_score(y_test, nb_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("\nAccuracy Scores")
print("----------------------------")
print("Logistic Regression :", lr_acc)
print("Naive Bayes         :", nb_acc)
print("Random Forest       :", rf_acc)

# ============================================
# Classification Report
# ============================================

print("\nRandom Forest Classification Report")
print(classification_report(y_test, rf_pred))

# ============================================
# Confusion Matrix
# ============================================

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("images/confusion_matrix.png")
plt.show()

# ============================================
# Accuracy Graph
# ============================================

models = [
    "Logistic Regression",
    "Naive Bayes",
    "Random Forest"
]

accuracies = [
    lr_acc,
    nb_acc,
    rf_acc
]

plt.figure(figsize=(7,5))
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.90, 1.01)
plt.savefig("images/accuracy_comparison.png")
plt.close()

# ============================================
# Save Best Model
# ============================================

joblib.dump(rf, "iris_model.pkl")
print("\nRandom Forest model saved successfully!")

# ============================================
# Prediction
# ============================================

sample = np.array([[5.1, 3.5, 1.4, 0.2]])

prediction = rf.predict(sample)

print("\nPredicted Flower:")
print(iris.target_names[prediction][0])