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

# --- FUNGSI UPLOAD TAHAN BANTING ---
def upload_with_retry(local_file, artifact_path, max_retries=3):
    """Mencoba upload file, jika gagal (Error 500), tunggu lalu coba lagi."""
    for i in range(max_retries):
        try:
            print(f"Attempt {i+1}: Uploading {os.path.basename(local_file)}...")
            mlflow.log_artifact(local_file, artifact_path=artifact_path)
            print("Sukses.")
            return # Keluar fungsi jika sukses
        except Exception as e:
            print(f"Gagal upload (Error: {str(e)})")
            if i < max_retries - 1:
                wait_time = 15 * (i + 1) # Tunggu 15s, lalu 30s...
                print(f"Server sibuk. Menunggu {wait_time} detik sebelum retry...")
                time.sleep(wait_time)
            else:
                print("Menyerah setelah 3x percobaan.")
                raise e # Lemparkan error jika sudah mentok

# --- 1. KONFIGURASI OTOMATIS ---
in_ci_cd = os.getenv("MLFLOW_TRACKING_URI") is not None

if in_ci_cd:
    print("Mode: CI/CD (GitHub Actions)")
    run_context = mlflow.start_run() 
else:
    print("Mode: Lokal (Laptop)")
    dagshub.init(repo_owner='amirmahmoed003', repo_name='proyek_akhir_msml_amir', mlflow=True)
    mlflow.set_experiment("Proyek_Akhir_CI_CD")
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

# --- 3. TRAINING & UPLOAD ---
with run_context as run:
    
    mlflow.set_tag("mlflow.runName", "CI_CD_Model_Final")
    
    # Simpan Run ID
    run_id = run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
        
    print(f"Run ID: {run_id}")
    print("Autolog dimatikan. Memulai training manual...")

    # Training
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc}")
    
    # Log Metric Manual
    mlflow.log_metric("accuracy", acc)
    mlflow.log_params({"n_estimators": 100, "lr": 0.1, "depth": 3})
    
    # --- PROSES UPLOAD ---
    print("Menyiapkan model lokal...")
    local_model_path = "temp_model_dir"
    
    if os.path.exists(local_model_path):
        shutil.rmtree(local_model_path)
    
    mlflow.sklearn.save_model(model, local_model_path)
    
    print("Mulai mengupload file model dengan Smart Retry...")
    
    for root, dirs, files in os.walk(local_model_path):
        for filename in files:
            local_file = os.path.join(root, filename)
            
            # Tentukan path tujuan
            relative_path = os.path.relpath(local_file, local_model_path)
            dir_name = os.path.dirname(relative_path)
            
            if dir_name:
                dest_path = os.path.join("model", dir_name)
            else:
                dest_path = "model"
            
            # PANGGIL FUNGSI RETRY DI SINI
            upload_with_retry(local_file, dest_path)
            
            # Jeda standar antar file
            time.sleep(5)
            
    print("Semua file model berhasil diupload!")
    shutil.rmtree(local_model_path)
    
    # --- ARTEFAK VISUALISASI ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    upload_with_retry("confusion_matrix.png", None)

    print("Selesai! CI/CD Sukses.")