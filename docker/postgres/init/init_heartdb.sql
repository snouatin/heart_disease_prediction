-- Création de la table heart_data
DROP TABLE IF EXISTS heart_data;
CREATE TABLE heart_data (
    id             SERIAL PRIMARY KEY,
    age            INTEGER,
    sex            INTEGER,
    chest_pain_type INTEGER,
    resting_bp     INTEGER,
    cholesterol    INTEGER,
    fasting_bs     INTEGER,
    rest_ecg       INTEGER,
    max_heart_rate INTEGER,
    exercise_angina INTEGER,
    oldpeak        NUMERIC,
    st_slope       INTEGER,
    target         INTEGER
);

-- Import des données depuis le CSV monté dans /docker-entrypoint-initdb.d/
COPY heart_data(
    age,
    sex,
    chest_pain_type,
    resting_bp,
    cholesterol,
    fasting_bs,
    rest_ecg,
    max_heart_rate,
    exercise_angina,
    oldpeak,
    st_slope,
    target
)
FROM '/docker-entrypoint-initdb.d/heart.csv'
DELIMITER ',' CSV HEADER;
