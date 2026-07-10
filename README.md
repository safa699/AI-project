# AI-project

# 🌸 Iris Flower Species Recognition System using Machine Learning

## 🚀 About The Project

This project focuses on building an intelligent Machine Learning system that can automatically recognize Iris flower species by analyzing their physical characteristics.

A complete ML workflow is implemented, including data preprocessing, model training, performance evaluation, and deployment. Multiple classification algorithms are tested, and the most effective model is integrated into a Flask web application for real-time predictions.

---

## 📚 Dataset Information

The model is trained using the **Iris Flower Dataset** from Scikit-learn, a popular dataset used for classification problems in Machine Learning.

### Flower Measurements Used:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

### Output Classes:

The system predicts one of the following flower species:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

---

## 🧠 Machine Learning Models Implemented

To achieve accurate classification, different supervised learning algorithms were trained and compared:

🔹 Logistic Regression
🔹 Gaussian Naive Bayes
🔹 Random Forest Algorithm

After analyzing the performance of each model, the Random Forest Classifier was selected as the final model because of its strong prediction accuracy.

---

## 📈 Analysis & Performance Visualization

The project provides visual analysis of the dataset and model performance using:

* Pairwise Feature Analysis
* Correlation Heatmap
* Confusion Matrix Evaluation
* Accuracy Comparison Visualization

These graphs help in understanding relationships between features and measuring model effectiveness.

---

## 🌐 Web-Based Prediction Application

A Flask-powered web interface is developed to make the prediction system easy to use.

### Application Features:

✅ Enter flower measurements manually
✅ Generate instant species predictions
✅ Connect with trained ML model
✅ Simple and interactive interface

---

## 💻 Tools & Technologies

**Programming Language**

* Python

**Framework**

* Flask

**Machine Learning**

* Scikit-learn

**Data Processing**

* Pandas
* NumPy

**Visualization**

* Matplotlib
* Seaborn

**Model Saving**

* Joblib

---

## 📂 Directory Layout

```text
Iris-Flower-Classification/

│── app.py
│── model_training.py
│── iris_model.pkl

│── static/
│── templates/
│── images/

│── README.md
```

---

## ⚙️ Installation & Execution

### Install Required Packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn flask joblib
```

### Train the Model

```bash
python model_training.py
```

### Launch Web Application

```bash
python app.py
```

Open the following address:

```text
http://127.0.0.1:5000
```

---

## 🏆 Project Outcome

The implemented classification models provided highly accurate results on the Iris dataset.

The Random Forest Classifier was finalized as the prediction engine for the Flask application due to its reliable classification performance.

---

## 👩‍💻 Developed By

**Safa Malik**
BS Computer Science Student
Machine Learning | Python | Web Development
