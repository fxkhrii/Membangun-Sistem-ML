FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir mlflow==2.19.0 scikit-learn pandas numpy

# Copy model artifacts
COPY model/ /app/model/

# Expose port for serving
EXPOSE 5001

# Serve the model
CMD ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "5001", "--no-conda"]
