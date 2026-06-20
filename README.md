# 🏦 Home Credit Default Risk - MLOps Project

A complete end-to-end Machine Learning and MLOps system for predicting loan default risk using Logistic Regression, Random Forest, XGBoost, FastAPI, Streamlit, Docker, and MLflow.

🌐 Live Demo: https://home-credit-default-risk-mlops.onrender.com

---

## 🚀 Overview

This project predicts whether a loan applicant is likely to default on a loan based on financial and behavioral features.

### Features

- Data preprocessing
- Class imbalance handling with SMOTE
- Logistic Regression
- Random Forest
- XGBoost
- MLflow Experiment Tracking
- MLflow Model Registry
- FastAPI REST API
- Streamlit Dashboard
- Docker Deployment
- Render Cloud Deployment

---

## 🧠 Problem Statement

Banks need to assess credit risk accurately to reduce loan defaults and financial losses.

**Goal:** Predict whether a customer will default on a loan.

- `0` = No Default
- `1` = Default

---

## 🏗️ Architecture

```text
Data
 ↓
Preprocessing
 ↓
SMOTE
 ↓
Model Training
 ↓
Evaluation
 ↓
MLflow Tracking
 ↓
Model Registry
 ↓
FastAPI
 ↓
Streamlit
 ↓
Docker
 ↓
Render Deployment
```

---

## ⚙️ Tech Stack

### Machine Learning

- Python
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Pandas
- NumPy

### MLOps

- MLflow
- Joblib

### Backend

- FastAPI
- Uvicorn

### Frontend

- Streamlit

### DevOps

- Docker
- Docker Compose

### Deployment

- Render

---

## 📁 Project Structure

```text
Home-Credit-Default-Risk-MLOps/
│
├── src/
│   ├── train.py
│   ├── api.py
│   ├── data_preprocessing.py
│   └── config.py
│
├── models/
│   ├── LogisticRegression.joblib
│   ├── RandomForestClassifier.joblib
│   └── XGBClassifier.joblib
│
├── tests/
├── streamlit_app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── mlflow.db
├── README.md
└── logs/
```

---

## 🧪 Model Training Pipeline

1. Load dataset
2. Preprocess features
3. Handle imbalance with SMOTE
4. Train models:
   - Logistic Regression
   - Random Forest
   - XGBoost
5. Evaluate performance
6. Log experiments with MLflow
7. Register models in MLflow Registry
8. Save trained pipelines using Joblib

---

## 📊 Model Comparison

| Model | Accuracy | ROC-AUC | F1 Score | Recall | Precision |
|---------|---------:|---------:|---------:|---------:|---------:|
| Logistic Regression | 0.695 | 0.743 | 0.260 | 0.663 | 0.161 |
| Random Forest | 0.840 | 0.665 | 0.213 | 0.268 | 0.176 |
| XGBoost | 0.919 | 0.734 | 0.022 | 0.011 | 0.539 |

**Note:** Due to class imbalance, ROC-AUC, Recall, and F1 Score are more informative than Accuracy.

---

## 📈 MLflow Tracking

Run MLflow locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

```text
http://127.0.0.1:5000
```

Registered Models:

- home_credit_lr
- home_credit_rf
- home_credit_xgb

---

## 🌐 FastAPI Backend

Run locally:

```bash
uvicorn src.api:app --reload
```

Endpoint:

```http
POST /predict
```

Example request:

```json
{
  "features": [...]
}
```

Example response:

```json
{
  "prediction": 1,
  "probability": 0.87
}
```

---

## 🎨 Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

---

## 🐳 Docker

Build:

```bash
docker build -t home-credit-api .
```

Run:

```bash
docker run -p 8000:8000 home-credit-api
```

---

## ✅ Testing

```bash
pytest
```

```bash
python -m src.train
```

```bash
uvicorn src.api:app --reload
```

```bash
streamlit run streamlit_app.py
```

```bash
docker run -p 8000:8000 home-credit-api
```

---

## ☁️ Deployment

Live Application:

https://home-credit-default-risk-mlops.onrender.com

---

## 👨‍💻 Author

**Brahim AZREG**

Machine Learning & MLOps Engineer