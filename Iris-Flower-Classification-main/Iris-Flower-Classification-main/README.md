# 🌸 Iris Flower Classification using Machine Learning

##  Project Overview

This project is a Machine Learning application that classifies Iris flowers into three different species based on their measurements.

The application compares multiple machine learning algorithms and uses the best-performing model for prediction through a Flask web application.

---

## 🌼 Dataset

The project uses the famous **Iris Dataset** available in Scikit-learn.

Features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target Classes:
- Setosa
- Versicolor
- Virginica

---

##  Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Gaussian Naive Bayes
- Random Forest Classifier

---

## 📊 Visualizations

The project generates:

- Pair Plot
- Correlation Heatmap
- Confusion Matrix
- Accuracy Comparison Graph

---

##  Flask Web Application

The trained Random Forest model is deployed using Flask.

Users can:

- Enter flower measurements
- Click **Predict**
- View the predicted Iris species instantly

---

##  Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

---

##  Project Structure

```
Iris-Flower-Classification/

│── app.py
│── iris_classification.py
│── iris_model.pkl

│── images/
│── static/
│── templates/

│── README.md
```

---

##  How to Run

Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn flask joblib
```

Run the training script:

```bash
python iris_classification.py
```

Run the Flask application:

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

## 📈 Results

All three models achieved excellent performance on the Iris dataset.

The Random Forest Classifier was selected as the final prediction model.

---

##  Developed By

**Misbah Saeed**

BS Computer Science Student

Institute of Space Technology (IST)

---
