# Workflow CI - MLflow Training

Repository ini merupakan lanjutan dari eksperimen sistem machine learning (SML).
Workflow CI dibuat menggunakan GitHub Actions dan MLflow Project untuk melakukan
retraining model secara otomatis setiap terjadi push ke branch main.

## Struktur
- MLProject/
  - modelling.py
  - MLproject
  - conda.yaml
  - titanic_preprocessing/
- .github/workflows/ci.yml

## Workflow
- Trigger: push ke main
- Proses:
  - Setup Python
  - Install dependency
  - Jalankan `mlflow run`
  - Simpan artefak hasil training

## Tools
- MLflow
- Scikit-learn
- GitHub Actions
