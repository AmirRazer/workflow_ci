import pandas as pd
import mlflow
import dagshub
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. KONFIGURASI OTOMATIS 

# Cek apakah script berjalan di GitHub Actions (CI/CD)?
in_ci_cd = os.getenv("MLFLOW_TRACKING_URI") is not None

if in_ci_cd:
    print("Mode: CI/CD (GitHub Actions)")
    print("Menggunakan Token & URI dari Environment Variables.")
    # TIDAK perlu dagshub.init karena sudah diset di main.yml
    
    # Saat di CI/CD, MLflow sudah membuatkan Run otomatis.
    # cukup 'attach' (nempel) ke run tersebut tanpa memberi nama baru.
    run_context = mlflow.start_run() 
    
else:
    print("Mode: Lokal (Laptop)")
    print("Melakukan inisialisasi DagsHub manual...")
    dagshub.init(repo_owner='amirmahmoed003', repo_name='proyek_akhir_msml_amir', mlflow=True)
    
    # Jika di laptop,  perlu set nama experiment dan run sendiri
    mlflow.set_experiment("Proyek_Akhir_CI_CD")
    run_context = mlflow.start_run(run_name="Manual_Run_Lokal")

# --- 2. LOAD DATA ---
try:
    # Coba baca langsung (untuk struktur MLProject)
    X_train = pd.read_csv('train_data_processed.csv')
    X_test = pd.read_csv('test_data_processed.csv')
except FileNotFoundError:
    # Fallback path (jika dijalankan dari folder lain)
    X_train = pd.read_csv('../train_data_processed.csv')
    X_test = pd.read_csv('../test_data_processed.csv')

y_train = X_train.pop('Status')
y_test = X_test.pop('Status')

# --- 3. TRAINING DALAM CONTEXT RUN ---
# 'with run_context' akan otomatis memakai Run yang benar (baik dari GitHub maupun Lokal)
with run_context as run:
    
    # Simpan Run ID ke file 
    run_id = run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
        
    print(f"Run ID Aktif: {run_id}")
    
    # === BAGIAN AUTO ===
    print("Mengaktifkan Autolog...")
    mlflow.sklearn.autolog(log_models=False, exclusive=False)
    
    # Definisi Grid Search
    param_grid = {
        'n_estimators': [50, 100],     
        'learning_rate': [0.1], 
        'max_depth': [3]             
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    
    print("Memulai Training...")
    grid_search.fit(X_train, y_train)
    
    # === BAGIAN MANUAL & ARTEFAK ===
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Akurasi Terbaik: {acc}")

    # Log Metric Manual
    mlflow.log_metric("accuracy_manual", acc)
    
    # Log Model Standar (WAJIB ADA untuk Docker)
    print("Logging Model...")
    mlflow.sklearn.log_model(best_model, "model")
    
    # Artefak Visualisasi
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    print("Selesai! Run ID tersimpan.")