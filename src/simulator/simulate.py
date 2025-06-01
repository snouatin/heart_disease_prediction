import time
import json
import random
import uuid
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# Configuration Kafka
KAFKA_BROKER = 'kafka:9092'
TOPIC = 'patient-data'

def init_producer(retries=5, delay=5):
    """
    Tente de créer un KafkaProducer avec retry si le broker n'est pas dispo.
    """
    attempt = 0
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            )
            print("Connected to Kafka broker!")
            return producer
        except NoBrokersAvailable:
            attempt += 1
            print(f"Kafka broker not available (attempt {attempt}). Retrying in {delay}s...")
            if attempt >= retries:
                raise RuntimeError("Failed to connect to Kafka broker after multiple attempts")
            time.sleep(delay)

def generate_patient():

    """
    Génère un enregistrement patient synthétique.
    Champs basés sur le schéma Heart Disease UCI sans le label de classe.
    """
    return {
        "patient_id": str(uuid.uuid4()),
        "age": random.randint(29, 77),
        "sex": random.choice([0, 1]),  # 1=male, 0=female
        "chest_pain_type": random.randint(1, 4),  # 1-4
        "resting_bp": random.randint(94, 200),  # mm Hg
        "cholesterol": random.randint(126, 564),  # mg/dl
        "fasting_bs": random.choice([0, 1]),  # >120 mg/dl
        "rest_ecg": random.randint(0, 2),  # 0-2
        "max_heart_rate": random.randint(71, 202),
        "exercise_angina": random.choice([0, 1]),
        "oldpeak": round(random.uniform(0.0, 6.2), 1),
        "st_slope": random.randint(1, 3),  # 1-3
    }

if __name__ == '__main__':
    # Initialisation du producer avec retry
    print(">>> simulate.py starting up <<<")
    producer = init_producer(retries=10, delay=7)
    print("Démarrage du simulateur de patients...")
    while True:
        patient = generate_patient()
        producer.send(TOPIC, value=patient)
        producer.flush()
        print(f"Sent: {patient}")
        time.sleep(2)