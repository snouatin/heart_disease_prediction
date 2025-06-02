import os
import json
import time
import logging
import psycopg2
from kafka import KafkaConsumer
from psycopg2.extras import execute_values

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("archiver")

# Variables d’environnement
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
INPUT_TOPIC = os.getenv("INPUT_TOPIC", "predictions")

DB_URI = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME")
}

def get_connection():
    conn = psycopg2.connect(**DB_URI)
    conn.autocommit = True
    return conn

def create_table_if_not_exists(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                patient_id TEXT NOT NULL,
                prediction DOUBLE PRECISION NOT NULL,
                timestamp TIMESTAMP NOT NULL
            );
        """)
    logger.info("Table 'predictions' vérifiée/créée.")

def main():
    # 1. Connexion à PostgreSQL
    conn = get_connection()
    create_table_if_not_exists(conn)

    # 2. Kafka Consumer
    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='archiver-group'
    )
    logger.info(f"Listening on topic '{INPUT_TOPIC}' for predictions...")

    # 3. Boucle de consommation
    for message in consumer:
        record = message.value
        patient_id = record.get("patient_id")
        prediction = record.get("prediction")
        # timestamp renvoyé depuis le producteur (float epoch)
        ts = record.get("timestamp", time.time())
        # Convertir en objet TIMESTAMP PostgreSQL
        ts_pg = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))

        # 4. INSERT dans PostgreSQL
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions (patient_id, prediction, timestamp)
                    VALUES (%s, %s, %s)
                    """,
                    (patient_id, prediction, ts_pg)
                )
            logger.info(f"Archived: patient_id={patient_id}, prediction={prediction:.4f}, timestamp={ts_pg}")
        except Exception as e:
            logger.error(f"Erreur lors de l'insertion en DB : {e}")
            # Ne pas arrêter le consumer sur une erreur ponctuelle
            continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down archiver...\n")
        pass
