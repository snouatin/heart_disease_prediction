import os
import time
import json
import logging

from kafka import KafkaConsumer, KafkaProducer
import mlflow.pyfunc
from prometheus_client import start_http_server, Counter, Histogram
from prometheus_client.core import CollectorRegistry

# Logging configuration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predictor")

# Environment variables

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
INPUT_TOPIC = os.getenv("INPUT_TOPIC", "patient-data")
OUTPUT_TOPIC = os.getenv("OUTPUT_TOPIC", "predictions")
MODEL_URI = os.getenv("MODEL_URI", "models:/rf_heart_disease_model/1")
PROM_PORT = int(os.getenv("PROM_PORT", "8000"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Configuration MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Prometheus metrics

REQUEST_COUNT = Counter(
'prediction_requests_total', 'Total number of prediction requests processed'
)
REQUEST_LATENCY = Histogram(
'prediction_request_latency_seconds', 'Latency for prediction requests', buckets=(0.1, 0.5, 1, 2, 5)
)

def load_model(model_uri):
    logger.info(f"Loading model from {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("Model loaded successfully.")
    return model

def main():
    # Start Prometheus metrics server
    start_http_server(PROM_PORT)
    logger.info(f"Prometheus metrics available at [http://0.0.0.0:{PROM_PORT}](http://0.0.0.0:{PROM_PORT})")

    # Load model
    model = load_model(MODEL_URI)

    # Kafka Consumer
    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='predictor-group'
    )

    # Kafka Producer
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    logger.info(f"Listening on topic '{INPUT_TOPIC}' for incoming patient data...")

    for message in consumer:
        start_time = time.time()
        REQUEST_COUNT.inc()

        patient = message.value
        # Convert to DataFrame row for model
        import pandas as pd
        df = pd.DataFrame([patient])
        
        if 'patient_id' in df.columns:
            df = df.drop('patient_id', axis=1)

        # Perform prediction
        try:
            prediction = model.predict(df)
            prob = int(prediction[0]) if prediction.ndim == 1 else int(prediction[0][0])
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            continue

        # Prepare output message
        output = {
            "patient_id": patient.get("patient_id"),
            "prediction": prob,
            "timestamp": time.time()
        }
        # Send to OUTPUT_TOPIC
        producer.send(OUTPUT_TOPIC, value=output)
        producer.flush()

        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        logger.info(f"Processed patient_id={patient.get('patient_id')} with prediction={prob:.4f} in {latency:.3f}s")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down predictor...")
        pass
