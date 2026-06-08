import pandas as pd
import requests
import time
import random

# Konfigurasi
PROXY_URL = "http://localhost:8000/predict"
DATASET_PATH = "../Membangun_model/dataset_preprocessing/processed_data.csv"

def simulate_traffic():
    print("Memuat dataset untuk simulasi...")
    try:
        df = pd.read_csv(DATASET_PATH)
        # Drop target column if it exists
        if "RainTomorrow" in df.columns:
            df = df.drop("RainTomorrow", axis=1)
    except FileNotFoundError:
        print(f"Dataset tidak ditemukan di {DATASET_PATH}. Pastikan path benar.")
        return

    print("Memulai pengiriman simulasi traffic ke Proxy Exporter...")
    print("Tekan Ctrl+C untuk menghentikan.")
    
    # Ambil list kolom
    columns = list(df.columns)
    
    try:
        while True:
            # Pilih 1 row secara acak
            random_idx = random.randint(0, len(df) - 1)
            row_data = df.iloc[random_idx].tolist()
            
            # Format payload untuk MLflow model (dataframe_split)
            payload = {
                "dataframe_split": {
                    "columns": columns,
                    "data": [row_data]
                }
            }
            
            try:
                # Kirim request ke proxy
                response = requests.post(PROXY_URL, json=payload, timeout=5)
                if response.status_code == 200:
                    print(f"[SUCCESS] Prediksi: {response.json()}")
                else:
                    print(f"[ERROR] Kode {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"[CONNECTION ERROR] Gagal terhubung ke proxy: {e}")
            
            # Delay acak antara 0.5 hingga 2 detik agar terlihat natural di Grafana
            time.sleep(random.uniform(0.5, 2.0))
            
    except KeyboardInterrupt:
        print("\nSimulasi dihentikan oleh user.")

if __name__ == "__main__":
    simulate_traffic()
