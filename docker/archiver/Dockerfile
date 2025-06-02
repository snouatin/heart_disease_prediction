FROM python:3.10-slim

# Installer les dépendances nécessaires
RUN pip install --no-cache-dir kafka-python psycopg2-binary

# Créer le répertoire de travail
WORKDIR /app

# Copier le code
COPY src/archiver/consume_and_store.py ./

# Expose un port pour ajouter monitoring à l'avenir
EXPOSE 8500

# Commande par défaut au démarrage du container
CMD ["python", "consume_and_store.py"]
