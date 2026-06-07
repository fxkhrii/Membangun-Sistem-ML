import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import os

def train_advanced_model():
    # 1. Setup DagsHub Tracking URI (Untuk Kriteria Advanced)
    # TODO: Ganti URL di bawah ini dengan Tracking URI DagsHub Anda!
    # 1. Setup DagsHub Tracking URI menggunakan dagshub.init
    import dagshub
    dagshub.init(repo_owner='fxkhrii', repo_name='Membangun-Sistem-ML', mlflow=True)
    
    mlflow.set_experiment("Advanced_Modelling_Tuning_v2")

    # Load dataset
    df = pd.read_csv("dataset_preprocessing/processed_data.csv")
    X = df.drop("RainTomorrow", axis=1)
    y = df["RainTomorrow"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # KITA TIDAK MENGGUNAKAN AUTOLOG SESUAI SYARAT ADVANCED
    # mlflow.sklearn.autolog() # (DIMATIKAN)

    with mlflow.start_run(run_name="Tuning_RandomForest"):
        # 2. Hyperparameter Tuning (Syarat Skilled/Advanced)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Log parameter terbaik secara manual
        mlflow.log_params(grid_search.best_params_)
        
        # Prediksi
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # 3. Manual Logging Metrics (Syarat Skilled)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics_dict = {
            "accuracy_score": acc,
            "precision_score": prec,
            "recall_score": rec,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics_dict)
        
        print(f"Model Tuned Accuracy: {acc:.4f}")
        print(f"Best Params: {grid_search.best_params_}")
        
        # 4. Generate Artifacts sesuai struktur Skilled
        import json
        from sklearn.utils import estimator_html_repr
        
        # metric_info.json
        with open("metric_info.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)
        mlflow.log_artifact("metric_info.json")
        
        # estimator.html
        with open("estimator.html", "w", encoding="utf-8") as f:
            f.write(estimator_html_repr(best_model))
        mlflow.log_artifact("estimator.html")
        
        # training_confusion_matrix.png
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("training_confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("training_confusion_matrix.png")
        
        # 5. Tambahan 2 Artefak Ekstra (Syarat Advanced: autolog + minimal 2 artefak)
        # Artefak Ekstra 1: ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig("roc_curve_advanced.png")
        plt.close()
        mlflow.log_artifact("roc_curve_advanced.png")
        
        # Artefak Ekstra 2: Feature Importance
        importances = best_model.feature_importances_
        plt.figure(figsize=(8,6))
        plt.barh(X.columns, importances)
        plt.title('Feature Importances')
        plt.savefig("feature_importances_advanced.png")
        plt.close()
        mlflow.log_artifact("feature_importances_advanced.png")
        
        # Log Model ke dalam folder 'model'
        mlflow.sklearn.log_model(best_model, "model")
        print("Training dan Logging Selesai! Cek MLflow UI / DagsHub Anda.")

if __name__ == "__main__":
    train_advanced_model()
