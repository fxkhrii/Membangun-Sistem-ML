import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import sys

def train_model():
    # Load dataset
    df = pd.read_csv("dataset_preprocessing/processed_data.csv")
    
    # Pisahkan fitur dan target
    X = df.drop("RainTomorrow", axis=1)
    y = df["RainTomorrow"]
    
    # Split train dan test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set MLflow tracking ke lokal (untuk CI)
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("CI_Modelling_SML")
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="CI_RandomForest") as run:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        score = model.score(X_test, y_test)
        print(f"CI Model Accuracy: {score:.4f}")
        print(f"MLflow Run ID: {run.info.run_id}")
        
        # Log model explicitly to ensure 'model' artifact folder exists
        mlflow.sklearn.log_model(model, "model")
        
        # Simpan run_id ke file agar bisa diambil oleh CI step berikutnya
        with open("latest_run_id.txt", "w") as f:
            f.write(run.info.run_id)

if __name__ == "__main__":
    train_model()
