# Utilisez une image de base avec support Jupyter et VSCode
FROM python:3.10

# Définissez le répertoire de travail
WORKDIR /sol

# Copiez les fichiers de votre projet dans l'image
COPY requirements.txt .
COPY ./schedule_optimizer ./schedule_optimizer

# Installez les dépendances
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Commande à exécuter lorsque le conteneur démarre
CMD ["python", "./scripts/main.py"]