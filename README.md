# ⚙️ System Health & Failure Predictor (ML + Backend Focused)

A production-oriented Machine Learning system that predicts machine failure risk using operational metrics and exposes predictions via a scalable REST API.

---

## 🌐 Live API

🔗 API Docs (Swagger UI):
https://system-health-failure-predictor.onrender.com/docs

---

## 🧠 Problem Statement

In industrial and system environments, unexpected failures can lead to:

* System downtime
* Financial loss
* Maintenance overhead

This project focuses on building a **predictive maintenance system** that:

👉 Estimates probability of machine failure
👉 Classifies risk into actionable categories
👉 Operates as a deployable backend service

---

## 📊 Dataset

🔗 https://www.kaggle.com/datasets/abdelazizsami/predictive-maintenance-dataset

### Features Used:

* Type (encoded: L/M/H → 0/1/2)
* Air Temperature [K]
* Process Temperature [K]
* Rotational Speed [rpm]
* Torque [Nm]
* Tool Wear [min]

### Target:

* Machine Failure (0 / 1)

---

## ⚠️ Data Engineering Decisions

### 1️⃣ Removed Data Leakage Columns

Dropped:

* TWF, HDF, PWF, OSF, RNF

👉 These directly indicate failure → would lead to unrealistic model performance

---

### 2️⃣ Feature Selection

Removed:

* UDI (index-like)
* Product ID (identifier)

---

### 3️⃣ Encoding

* Type column encoded using Label Encoding

---

### 4️⃣ Class Imbalance Handling

Dataset distribution:

* Class 0: ~97%
* Class 1: ~3%

Handled using:

```python
class_weight={0:1, 1:3}
```

---

## 🧪 Model Development

### Algorithms Evaluated:

* Logistic Regression
* Random Forest Classifier

---

### Final Model:

**Random Forest Classifier**

Reasons:

* Better balance between precision and recall
* Handles non-linear relationships
* Robust to feature interactions

---

## 📈 Model Performance

Key focus: **Recall for failure detection**

Example metrics:

* Precision (Failure): ~0.63
* Recall (Failure): ~0.79
* F1-score: ~0.70

👉 Model prioritizes **detecting failures over false safety**

---

## 🎯 Threshold Optimization

Instead of default threshold (0.5):

```python
y_pred = (probs > 0.3).astype(int)
```

👉 Improves recall for minority class (failures)

---

## ⚙️ Prediction Logic

Model output enhanced with interpretation layer:

```json
{
  "failure_probability": 0.79,
  "risk_level": "HIGH"
}
```

### Risk Mapping:

* < 0.3 → LOW
* 0.3 – 0.7 → MEDIUM
* > 0.7 → HIGH

---

## 🏗️ Backend Architecture

```text
Client Request
     ↓
FastAPI Endpoint (/predict)
     ↓
Input Validation (Pydantic)
     ↓
ML Model (.pkl)
     ↓
Probability + Risk Mapping
     ↓
JSON Response
```

---

## 🔌 API Design

### POST /predict

#### Request:

```json
{
  "Type": 1,
  "Air_temp": 298.3,
  "Process_temp": 308.5,
  "Rotational_speed": 1500,
  "Torque": 40.5,
  "Tool_wear": 80
}
```

#### Response:

```json
{
  "failure_probability": 0.72,
  "risk_level": "HIGH"
}
```

---

## ⚙️ Tech Stack

### Machine Learning

* Python
* scikit-learn
* Pandas
* NumPy

### Backend

* FastAPI
* Pydantic (validation)

### Deployment

* Render (cloud hosting)
* Docker (containerization)

---

## 🐳 Containerization

Docker used for:

* Environment consistency
* Dependency isolation
* Deployment portability

---

## ⚠️ Engineering Challenges

* Preventing data leakage
* Handling extreme class imbalance
* Tuning threshold for real-world relevance
* Designing clean API interface
* Debugging CORS issues for integration
* Docker build failures (dependency + path issues)

---

## 📌 Key Learnings

* ML systems must prioritize **real-world impact over accuracy**
* Handling imbalance is critical in anomaly detection problems
* Threshold tuning can outperform default model behavior
* Data leakage can invalidate entire models
* Backend integration is essential for ML production systems

---

## 🚀 Future Improvements

* Feature importance explanations (SHAP / LIME)
* Model monitoring & logging
* Auto-retraining pipeline
* Batch prediction support
* Authentication for API usage

---

## 👨‍💻 Author

**Nishot Timalsina**

---
