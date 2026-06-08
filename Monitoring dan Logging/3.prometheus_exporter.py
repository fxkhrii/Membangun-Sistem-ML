from fastapi import FastAPI, HTTPException, Request
import requests
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="ML Model Prometheus Exporter")

# MLflow model endpoint
MLFLOW_MODEL_URL = "http://localhost:5001/invocations"

# ----------------- PROMETHEUS METRICS -----------------

# 1. Total Requests (Counter)
REQUEST_COUNT = Counter(
    "model_requests_total",
    "Total requests received by the proxy"
)

# 2. Successful Requests (Counter)
SUCCESS_COUNT = Counter(
    "model_requests_successful",
    "Total successful requests"
)

# 3. Failed Requests (Counter)
FAILED_COUNT = Counter(
    "model_requests_failed",
    "Total failed requests"
)

# 4. Latency (Histogram)
LATENCY_HISTOGRAM = Histogram(
    "model_latency_seconds",
    "Time taken for the model to respond",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# 5. Prediction Classes (Counter)
PREDICTION_CLASS_TOTAL = Counter(
    "model_prediction_class_total",
    "Total predictions per class",
    ["prediction"]
)

# 6. MinTemp (Histogram)
MINTEMP_HISTOGRAM = Histogram(
    "model_input_min_temp",
    "Distribution of MinTemp input",
    buckets=[-10, 0, 10, 20, 30, 40]
)

# 7. MaxTemp (Histogram)
MAXTEMP_HISTOGRAM = Histogram(
    "model_input_max_temp",
    "Distribution of MaxTemp input",
    buckets=[-10, 0, 10, 20, 30, 40, 50]
)

# 8. Rainfall (Histogram)
RAINFALL_HISTOGRAM = Histogram(
    "model_input_rainfall",
    "Distribution of Rainfall input",
    buckets=[0, 2, 5, 10, 20, 50, 100]
)

# 9. Humidity9am (Histogram)
HUMIDITY9AM_HISTOGRAM = Histogram(
    "model_input_humidity_9am",
    "Distribution of Humidity9am input",
    buckets=[0, 20, 40, 60, 80, 100]
)

# 10. Humidity3pm (Histogram)
HUMIDITY3PM_HISTOGRAM = Histogram(
    "model_input_humidity_3pm",
    "Distribution of Humidity3pm input",
    buckets=[0, 20, 40, 60, 80, 100]
)

# -----------------------------------------------------

@app.get("/metrics")
def metrics():
    """Endpoint untuk disekrap oleh Prometheus"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
async def predict(request: Request):
    """
    Endpoint proxy:
    1. Menerima request user
    2. Mencatat metrik (Input features)
    3. Meneruskan ke MLflow server
    4. Mencatat metrik (Output / Latency)
    5. Mengembalikan response ke user
    """
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    try:
        # Baca payload dari user
        payload = await request.json()
        
        # Ekstrak fitur untuk metrik (asumsi payload memiliki format split orient atau dict record)
        # MLflow melayani format {'dataframe_split': {'columns': [...], 'data': [[...]]}} 
        # atau {'dataframe_records': [{...}]}
        
        # Kita parse payload secara sederhana jika berbentuk record untuk metrik
        # Kita letakkan di blok try-except agar jika formatnya tidak standar tidak error
        try:
            if "dataframe_records" in payload:
                record = payload["dataframe_records"][0]
                if "MinTemp" in record: MINTEMP_HISTOGRAM.observe(record["MinTemp"])
                if "MaxTemp" in record: MAXTEMP_HISTOGRAM.observe(record["MaxTemp"])
                if "Rainfall" in record: RAINFALL_HISTOGRAM.observe(record["Rainfall"])
                if "Humidity9am" in record: HUMIDITY9AM_HISTOGRAM.observe(record["Humidity9am"])
                if "Humidity3pm" in record: HUMIDITY3PM_HISTOGRAM.observe(record["Humidity3pm"])
            elif "dataframe_split" in payload:
                cols = payload["dataframe_split"]["columns"]
                data = payload["dataframe_split"]["data"][0]
                record = dict(zip(cols, data))
                if "MinTemp" in record: MINTEMP_HISTOGRAM.observe(record["MinTemp"])
                if "MaxTemp" in record: MAXTEMP_HISTOGRAM.observe(record["MaxTemp"])
                if "Rainfall" in record: RAINFALL_HISTOGRAM.observe(record["Rainfall"])
                if "Humidity9am" in record: HUMIDITY9AM_HISTOGRAM.observe(record["Humidity9am"])
                if "Humidity3pm" in record: HUMIDITY3PM_HISTOGRAM.observe(record["Humidity3pm"])
        except Exception as e:
            print(f"Failed to parse payload for metrics: {e}")
        
        # Teruskan ke MLflow Model Serve
        response = requests.post(MLFLOW_MODEL_URL, json=payload, headers={'Content-Type': 'application/json'}, timeout=10)
        
        # Record latency
        latency = time.time() - start_time
        LATENCY_HISTOGRAM.observe(latency)
        
        if response.status_code == 200:
            SUCCESS_COUNT.inc()
            result = response.json()
            
            # Ekstrak prediksi untuk metrik
            try:
                # result biasanya berbentuk {'predictions': [0]} atau `[0]`
                preds = result.get('predictions', result) if isinstance(result, dict) else result
                if isinstance(preds, list) and len(preds) > 0:
                    pred_class = str(preds[0])
                    PREDICTION_CLASS_TOTAL.labels(prediction=pred_class).inc()
            except Exception as e:
                print(f"Failed to parse prediction for metrics: {e}")
                
            return result
        else:
            FAILED_COUNT.inc()
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except requests.exceptions.RequestException as e:
        FAILED_COUNT.inc()
        raise HTTPException(status_code=500, detail=f"Failed to connect to MLflow server: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Menjalankan exporter ini di port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
