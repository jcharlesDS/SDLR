import os
from dotenv import load_dotenv
from pathlib import Path

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Chemins de base
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR = STATIC_DIR / "audio"
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"

# Créer les dossiers s'ils n'existent pas
UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)
TRAINED_MODELS_DIR.mkdir(exist_ok=True)

# Langues supportées
SUPPORTED_LANGUAGES = ["fr", "en"]
LANGUAGE_NAMES = {
    "fr": "Français",
    "en": "Anglais"
}

# Configuration audio
AUDIO_CONFIG = {
    "sample_rate": 16000, # Fréquence d'échantillonnage standard pour la reconnaissance vocale
    "max_duration": 3600, # Durée maximale des fichiers audio en secondes (1 heure)
    "allowed_formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a"], # Formats audio supportés
    "max_file_size": 500 * 1024 * 1024 # Taille maximale des fichiers audio en octets (500 Mo)
}

# Configuration des modèles (Whisper + pyannote)
MODEL_CONFIG = {
    "whisper": {
        "model_size": "medium",  # medium (769M) pour meilleure qualité - plus lent que small (244M) mais plus précis
        "device": "cpu",  # cpu par défaut, cuda si l'utilisateur le demande
        "language": None,  # None = auto-detect, ou "fr"/"en" pour forcer
    },
    "pyannote": {
        "pipeline": "pyannote/speaker-diarization-3.1",
        "device": "cpu",  # cpu par défaut, cuda si l'utilisateur le demande
        "auth_token": os.getenv("HUGGINGFACE_TOKEN"),  # Nécessaire pour pyannote (HuggingFace token)
        "min_speakers": None,  # None = auto, ou nombre min de locuteurs
        "max_speakers": None,  # None = auto, ou nombre max de locuteurs
    },
    "transcription": {
        "enabled": False,  # Transcription optionnelle (sera activée par l'utilisateur)
        "combine_with_diarization": True,  # Combiner transcription + diarization
    }
}

# Configuration FastAPI
API_CONFIG = {
    "title": "Speaker Diarization & Language Recognition",
    "description": "API pour l'analyse audio : détection de langue et séparation des locuteurs",
    "version": "1.0.0",
    "host": "0.0.0.0",
    "port": 8000
}