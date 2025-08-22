# 🚢 Titanic Survival Prediction API

A simple **FastAPI** service that predicts survival probability on the Titanic dataset using a **RandomForestClassifier**.  
Preprocessing and model training are encapsulated in a `Pipeline`, ensuring no data leakage.

---

## 📂 Files

- `train.py` → trains the model and saves `titanic_model.joblib`
- `app.py` → FastAPI service exposing `/predict` and `/health`
- `requirements.txt` → dependencies
- `Dockerfile` → containerized deployment
- `titanic_random_forresst.py` → (previous notebook/script you had)

---

## ⚡ Run locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
