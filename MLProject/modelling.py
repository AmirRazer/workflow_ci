import pandas as pd
import mlflow
import dagshub
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil # <--- TAMBAHAN PENTING
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. KONFIGURASI OTOMATIS ---
in_ci_cd = os.getenv("MLFLOW_TRACKING_URI") is not None

if in_ci_cd:
    print("Mode: CI/CD (GitHub Actions)")
    # Attach ke run yang sudah dibuat oleh GitHub Actions
    run_context = mlflow.start_run() 
else:
    print("Mode: Lokal (Laptop)")
    dagshub.init(repo_owner='amirmahmoed003', repo_name='proyek_akhir_msml_amir', mlflow=True)
    mlflow.set_experiment("Proyek_Akhir_CI_CD")
    run_context = mlflow.start_run(run_name="Manual_Run_Lokal")

# --- 2. LOAD DATA ---
try:
    X_train = pd.read_csv('train_data_processed.csv')
    X_test = pd.read_csv('test_data_processed.csv')
except FileNotFoundError:
    X_train = pd.read_csv('../train_data_processed.csv')
    X_test = pd.read_csv('../test_data_processed.csv')

y_train = X_train.pop('Status')
y_test = X_test.pop('Status')

# --- 3. TRAINING ---
with run_context as run:
    
    # Simpan Run ID (Penting untuk Docker)
    run_id = run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
        
    print(f"Run ID Aktif: {run_id}")
    
    # Matikan log_models otomatis agar tidak error unsupported endpoint
    print("Mengaktifkan Autolog (Tanpa Model)...")
    mlflow.sklearn.autolog(log_models=False, exclusive=False)
    
    # Training
    param_grid = {
        'n_estimators': [50, 100],     
        'learning_rate': [0.1], 
        'max_depth': [3]             
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    
    print("Memulai Training...")
    grid_search.fit(X_train, y_train)
    
    # Evaluasi
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Akurasi Terbaik: {acc}")
    mlflow.log_metric("accuracy_manual", acc)
    
    # --- SOLUSI ERROR 'UNSUPPORTED ENDPOINT' ---
    print("Menyimpan Model dengan Metode Workaround...")
    
    # 1. Simpan model sebagai struktur MLflow di folder sementara LOKAL
    local_model_path = "temp_model_dir"
    if os.path.exists(local_model_path):
        shutil.rmtree(local_model_path)
        
    mlflow.sklearn.save_model(best_model, local_model_path)
    
    # 2. Upload folder tersebut sebagai ARTEFAK BIASA
    # Ini membypass API endpoint log_model yang error di DagsHub
    mlflow.log_artifacts(local_model_path, artifact_path="model")
    
    # 3. Hapus folder sementara
    shutil.rmtree(local_model_path)
    print("Model berhasil diupload sebagai Artefak MLflow.")
    
    # --- Artefak Lain ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    print("Selesai! Run ID tersimpan.")