-- Création de la table predictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    patient_id TEXT NOT NULL,
    prediction DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMP NOT NULL
);
