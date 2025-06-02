FROM python:3.10-slim

# Install dependencies

RUN pip install --no-cache-dir kafka-python mlflow pandas prometheus-client

WORKDIR /app

# Copy application code

COPY src/predictor/consume_and_predict.py ./

# Expose Prometheus metrics port

EXPOSE 8000

# Start predictor

CMD ["python", "consume_and_predict.py"]
