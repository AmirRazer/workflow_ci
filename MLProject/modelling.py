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
    
    # --- STRATEGI BARU: ZIP UPLOAD & KEEP LOCAL ---
    print("Menyiapkan model lokal...")
    
    # Folder tempat model disimpan (JANGAN DIHAPUS nanti, biar dipakai Docker)
    output_model_dir = "final_model"
    
    if os.path.exists(output_model_dir):
        shutil.rmtree(output_model_dir)
    
    # 1. Generate struktur model MLflow di folder 'final_model'
    mlflow.sklearn.save_model(model, output_model_dir)
    
    # 2. ZIP folder tersebut untuk diupload ke DagsHub (Solusi Anti Error 500)
    print("Membuat arsip ZIP untuk DagsHub...")
    shutil.make_archive("model_archive", 'zip', output_model_dir)
    
    # 3. Upload ZIP-nya saja (1 File, Cepat & Sukses)
    print("Mengupload model_archive.zip ke DagsHub...")
    try:
        mlflow.log_artifact("model_archive.zip", artifact_path="model_backup")
        print("Upload ZIP Sukses!")
    except Exception as e:
        print(f"Warning: Gagal upload ke DagsHub ({e}), tapi proses akan lanjut untuk Docker.")

    # Catatan: Kita TIDAK menghapus folder 'final_model' agar bisa dibaca step Docker di main.yml
    print("Folder model lokal 'final_model' siap untuk Docker Build.")
    
    # --- ARTEFAK VISUALISASI ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    try:
        mlflow.log_artifact("confusion_matrix.png")
    except:
        pass

    print("Selesai! CI/CD Sukses.")