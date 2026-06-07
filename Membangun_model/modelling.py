import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_basic_model():
    # Load dataset
    df = pd.read_csv("dataset_preprocessing/processed_data.csv")
    
    # Pisahkan fitur dan target (asumsi target adalah kolom 'RainTomorrow' berdasarkan preprocessing sebelumnya)
    X = df.drop("RainTomorrow", axis=1)
    y = df["RainTomorrow"]
    
    # Split train dan test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Aktifkan autologging MLflow untuk kriteria Basic
    # Menyimpan ke localhost atau 127.0.0.1 sesuai instruksi
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Basic_Modelling_SML")
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="Basic_RandomForest"):
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        score = model.score(X_test, y_test)
        print(f"Basic Model Accuracy: {score:.4f}")

if __name__ == "__main__":
    train_basic_model()
