# Base image avec Python 3.9
FROM python:3.9-slim

# Métadonnées
LABEL maintainer="mounirlamssiyah@gmail.com"
LABEL description="Entity Matching Siamese Neural Network"
LABEL version="1.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_PATH=/app/models
ENV DATA_PATH=/app/data

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Créer les répertoires nécessaires
RUN mkdir -p /app/src /app/models /app/data /app/outputs /app/logs

# Copier le code source
COPY src/ ./src/
COPY models/ ./models/

# Copier les fichiers de configuration
COPY *.py ./

# Permissions
RUN chmod +x src/*.py

# Port exposé (pour les APIs futures)
EXPOSE 5000

# Commande par défaut
CMD ["python", "src/app.py"]

# Pour le preprocessing
# CMD ["python", "src/data_preprocessing.py"]

# Pour l'entraînement  
# CMD ["python", "src/model_training.py"]