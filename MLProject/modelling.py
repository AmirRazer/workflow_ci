import pandas as pd
import mlflow
import dagshub
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. KONFIGURASI OTOMATIS ---
in_ci_cd = os.getenv("MLFLOW_TRACKING_URI") is not None

if in_ci_cd:
    print("Mode: CI/CD (GitHub Actions)")
    # Attach ke run yang dibuat GitHub Actions
    run_context = mlflow.start_run() 
else:
    print("Mode: Lokal (Laptop)")
    dagshub.init(repo_owner='amirmahmoed003', repo_name='proyek_akhir_msml_amir', mlflow=True)
    mlflow.set_experiment("Proyek_Akhir_CI_CD")
    # Nama run lokal
    run_context = mlflow.start_run(run_name="Run_Lokal_Laptop")

# --- 2. LOAD DATA ---
try:
    X_train = pd.read_csv('train_data_processed.csv')
    X_test = pd.read_csv('test_data_processed.csv')
except FileNotFoundError:
    X_train = pd.read_csv('../train_data_processed.csv')
    X_test = pd.read_csv('../test_data_processed.csv')

y_train = X_train.pop('Status')
y_test = X_test.pop('Status')

# --- 3. TRAINING (CEPAT & TANPA GRIDSEARCH) ---
with run_context as run:
    
    # [FIX NAMA RUN] Paksa ubah nama run agar terbaca jelas di DagsHub
    # Ini akan menimpa nama aneh seperti 'tasteful-cub-...'
    mlflow.set_tag("mlflow.runName", "CI_CD_Model_Final")
    
    # Simpan Run ID (Wajib untuk Docker)
    run_id = run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
        
    print(f"Run ID: {run_id} | Nama Run: CI_CD_Model_Final")
    
    # Matikan Autolog (Biar tidak error 500)
    print("Autolog dimatikan (Manual Log Only)...")
    
    # --- DEFINISI MODEL LANGSUNG (Tanpa GridSearch) ---
    # Kita pakai parameter yang umum saja biar cepat
    print("Melatih model Gradient Boosting (Single Model)...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    # Training cepat (cuma 1x proses)
    model.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Akurasi: {acc}")
    
    # Log Parameter & Metric Manual
    mlflow.log_params({"n_estimators": 100, "lr": 0.1, "depth": 3})
    mlflow.log_metric("accuracy", acc)
    
    # --- UPLOAD MODEL (WORKAROUND ANTI-ERROR) ---
    print("Menyiapkan upload model...")
    time.sleep(3) # Jeda sebentar
    
    # Simpan model struktur MLflow di lokal dulu
    local_model_path = "temp_model_dir"
    if os.path.exists(local_model_path):
        shutil.rmtree(local_model_path)
        
    mlflow.sklearn.save_model(model, local_model_path)
    
    print("Mengupload Model ke DagsHub...")
    # Upload folder model sebagai artefak
    mlflow.log_artifacts(local_model_path, artifact_path="model")
    
    # Bersihkan folder lokal
    shutil.rmtree(local_model_path)
    print("Model berhasil diupload.")
    
    # --- ARTEFAK VISUALISASI ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    mlflow.log_artifact("confusion_matrix.png")

    print("Selesai! Run ID tersimpan dan Nama Run sudah diperbaiki.")