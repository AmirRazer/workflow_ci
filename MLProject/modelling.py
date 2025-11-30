import pandas as pd
import mlflow
import dagshub
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. KONFIGURASI DAGSHUB ---
dagshub.init(repo_owner='amirmahmoed003', repo_name='proyek_akhir_msml_amir', mlflow=True)

# --- 2. LOAD DATA ---
X_train = pd.read_csv('train_data_processed.csv')
X_test = pd.read_csv('test_data_processed.csv')

y_train = X_train.pop('Status')
y_test = X_test.pop('Status')

# --- 3. TRAINING & LOGGING ---
mlflow.set_experiment("Proyek_Akhir_CI_CD")

with mlflow.start_run(run_name="CI_CD_Auto_git") as run:
    
    # [PENTING] Simpan Run ID ke file agar GitHub Actions bisa membacanya
    run_id = run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    
    # === BAGIAN AUTO ===
    print("Mengaktifkan Autolog...")
    # Kita matikan log_models di autolog agar tidak bentrok, kita log manual di bawah
    mlflow.sklearn.autolog(log_models=False, exclusive=False)
    
    # Definisi Grid Search
    param_grid = {
        'n_estimators': [50, 100],     
        'learning_rate': [0.1], # Dikurangi biar cepat di GitHub Actions 
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

    # 1. Log Metric Manual
    mlflow.log_metric("accuracy_manual", acc)
    
    # 2. Log Model Standar 
   
    print("Logging Model untuk Docker...")
    mlflow.sklearn.log_model(best_model, "model")
    
    # 3. Artefak Visualisasi
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    print("Selesai! Run ID tersimpan.")