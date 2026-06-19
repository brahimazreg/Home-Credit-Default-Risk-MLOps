# 🏦 Home Credit Default Risk - MLOps Project

A complete end-to-end Machine Learning + MLOps system for predicting loan default risk using Logistic Regression, FastAPI, Streamlit, Docker, and MLflow.

🌐 Live Demo: https://home-credit-default-risk-mlops.onrender.com

---

## 🚀 Overview

This project predicts whether a loan applicant will default based on financial and behavioral data.

It demonstrates a full MLOps pipeline:
- Data preprocessing
- Handling class imbalance (SMOTE)
- Model training (Logistic Regression)
- Experiment tracking (MLflow)
- API deployment (FastAPI)
- Frontend (Streamlit)
- Docker containerization
- Cloud deployment (Render)

---

## 🧠 Problem Statement

Banks need to assess credit risk accurately to reduce loan defaults and financial losses.

Goal:
> Predict probability of loan default (binary classification).

---

## 🏗️ Architecture

Data → Preprocessing → SMOTE → Training → MLflow Tracking → Model Saving → FastAPI → Streamlit → Docker → Deployment

---

## ⚙️ Tech Stack

**Machine Learning**
- Python
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Pandas, NumPy

**MLOps**
- MLflow (Experiment Tracking)
- Joblib (Model serialization)

**Backend**
- FastAPI
- Uvicorn

**Frontend**
- Streamlit

**DevOps**
- Docker
- Docker Compose

**Deployment**
- Render

---

### 🧠 Machine Learning
- Python
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Pandas / NumPy

### 📊 MLOps
- MLflow (Experiment Tracking)
- Joblib (Model serialization)

### 🌐 Backend
- FastAPI
- Uvicorn

### 🎨 Frontend
- Streamlit

### 🐳 DevOps
- Docker
- Docker Compose

### ☁️ Deployment
- Render (Cloud hosting)

---

## 📁 Project Structure

Home-Credit-Default-Risk-MLOps/
│
├── src/
│ ├── train.py
│ ├── api.py
│ ├── data_preprocessing.py
│ └── config.py
│
├── models/
│ └── logistic_regression.joblib
│
├── streamlit_app.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── tests/
└── logs/

## 🧪 Model Training Pipeline

Steps:
1. Load dataset
2. Preprocessing (encoding, scaling)
3. Handle imbalance using SMOTE
4. Train Logistic Regression model
5. Evaluate model:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
6. Log everything to MLflow
7. Save trained pipeline (`joblib`)

---

## 📊 MLflow Tracking

Run MLflow UI locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
then open : http://127.0.0.1:5000

FastAPI Backend
---------------

Run locally:
uvicorn src.api:app --reload

API endpoint:

POST /predict

Example request:

{
  "features": [...]
}

Response:

{
  "prediction": 1,
  "probability": 0.87
}

Streamlit UI
------------

Run locally:

streamlit run streamlit_app.py

Docker Setup
-------------

Build image:

docker build -t home-credit-api .

Run container:

docker run -p 8000:8000 home-credit-api


Deployment (Render)
------------------

The project is deployed on Render:

👉 https://home-credit-default-risk-mlops.onrender.com

Includes:

FastAPI backend
Docker container deployment
Production-ready API endpoint

👨‍💻 Author

Brahim AZREG