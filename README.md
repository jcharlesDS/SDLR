# Speaker Diarization & Language Recognition

> Application web FastAPI pour l'analyse audio intelligente : détection de langue, séparation des locuteurs et transcription automatique.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

---

## Table des matières

- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Technologies](#technologies)
- [Architecture](#architecture)

---

## Fonctionnalités

- **Détection automatique de langue** - Identification de la langue parlée avec score de confiance
- **Diarisation des locuteurs** - Séparation et identification automatique des différents locuteurs
- **Transcription audio** - Conversion parole-texte avec Whisper (optionnel)
- **Timeline interactive** - Visualisation des segments audio par locuteur avec couleurs
- **Lecteur audio intégré** - Navigation directe vers n'importe quel segment en cliquant sur les timestamps
- **Export de segments** - Extraction et téléchargement de segments audio spécifiques (MP3/WAV)
- **Mode batch** - Traitement de plusieurs fichiers simultanément (jusqu'à 10 fichiers)
- **Export multi-format** - Exportation des résultats en JSON, TXT ou SRT (sous-titres)
- **Mode sombre** - Interface moderne avec thème clair/sombre
- **Support GPU** - Accélération CUDA pour traitement rapide
- **Progression en temps réel** - Suivi détaillé de l'analyse via Server-Sent Events

---

## Prérequis

### Configuration système recommandée

- **GPU NVIDIA** avec support CUDA (optionnel mais fortement recommandé)
- **RAM** : 8 GB minimum (16 GB recommandé)
- **Fichiers audio** : Formats supportés `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg` (max 500 MB par fichier)

### Logiciels requis

- **Python** 3.9 ou supérieur
- **FFmpeg** (pour le traitement audio)

---

## Installation

### Étape 1 : Cloner le projet

```bash
git clone https://github.com/jcharlesDS/SDLR.git
cd SDLR
```

### Étape 2 : Créer l'environnement virtuel

**Windows (PowerShell) :**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS :**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Étape 3 : Installer PyTorch avec support CUDA

> **IMPORTANT** : PyTorch doit être installé AVANT les autres dépendances !

#### Option A : Carte graphique NVIDIA (recommandé)

1. Vérifiez votre modèle de GPU et sa compatibilité CUDA sur [nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus)

2. Installez PyTorch avec CUDA :

```bash
# Pour CUDA 12.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### Option B : CPU uniquement (plus lent)

```bash
pip install torch torchaudio
```

### Étape 4 : Installer FFmpeg

FFmpeg est requis pour le traitement des fichiers audio.

**Windows :**
- Téléchargez depuis [ffmpeg.org/download.html](https://www.ffmpeg.org/download.html)
- Extrayez et ajoutez le dossier `bin` à votre PATH système

**Linux (Ubuntu/Debian) :**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS :**
```bash
brew install ffmpeg
```

**Vérifier l'installation :**
```bash
ffmpeg -version
```

### Étape 5 : Installer les dépendances Python

```bash
pip install -r requirements.txt
```

---

## Configuration

### Configuration du token HuggingFace

Le modèle de diarisation PyAnnote nécessite un token HuggingFace.

#### Option 1 : Configuration via l'interface web (recommandé)

1. Lancez l'application (voir section [Utilisation](#utilisation))
2. Une fenêtre modale s'ouvrira automatiquement au premier lancement
3. Suivez les instructions à l'écran pour entrer votre token

#### Option 2 : Configuration manuelle

1. **Créer un compte HuggingFace** : [huggingface.co/join](https://huggingface.co/join)

2. **Accepter les conditions des modèles** :
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) (obligatoire)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (obligatoire)
   - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)(optionnel, mais parfois demandé)

3. **Générer un token** : [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Type : `Read` (lecture seule suffit)

4. **Créer un fichier `.env`** à la racine du projet :
   ```env
   HUGGINGFACE_TOKEN=hf_votre_token_ici
   ```

---

## Utilisation

### Démarrer l'application

```bash
# Mode développement (avec rechargement automatique)
python -m uvicorn app.main:app --reload

# Mode production
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Accéder à l'interface

Ouvrez votre navigateur : **[http://localhost:8000](http://localhost:8000)**

### Workflow d'analyse

1. **Upload** : Glissez-déposez ou sélectionnez un/des fichier(s) audio
2. **Configuration** :
   - Activer la transcription (optionnel)
   - Utiliser le GPU (si disponible)
   - Nombre de locuteurs (optionnel, détection auto sinon)
   - Fusion des segments proches (0.0 à 5.0 secondes)
3. **Analyser** : Cliquez sur "Analyser" et suivez la progression
4. **Résultats** : Consultez la timeline, transcription, et statistiques
5. **Actions** :
   - Cliquez sur un timestamp pour naviguer dans l'audio
   - Exportez des segments audio spécifiques
   - Exportez les résultats (JSON/TXT/SRT)

### Mode Batch

1. Cochez **"Mode Batch"** avant de sélectionner vos fichiers
2. Uploadez jusqu'à 10 fichiers simultanément
3. Naviguez entre les résultats avec les boutons ◀️ ▶️

---

## Technologies

### Backend
- **FastAPI** - Framework web moderne et rapide
- **PyTorch** - Deep learning et inférence des modèles
- **Whisper (OpenAI)** - Reconnaissance vocale state-of-the-art
- **PyAnnote** - Diarisation des locuteurs
- **Librosa** - Traitement et analyse audio
- **Pydub** - Manipulation de fichiers audio

### Frontend
- **HTML5/CSS3/JavaScript** - Interface responsive moderne
- **Server-Sent Events** - Progression en temps réel
- **Fetch API** - Communication asynchrone

### Modèles IA
- **Whisper** - Transcription multilingue
- **PyAnnote 3.1** - Diarisation des locuteurs
- **CNN custom** - Classification de langue (entraîné sur Common Voice)

---

## Architecture

```
Project_Speaker/
├── app/
│   ├── main.py              # Point d'entrée FastAPI
│   ├── models.py            # Modèles Pydantic
│   └── routes/
│       └── audio.py         # Endpoints API
├── models/
│   ├── diarization.py       # Diarisation PyAnnote
│   ├── language_id.py       # Détection de langue
│   ├── transcription.py     # Transcription Whisper
│   └── language_classifier.py  # CNN custom
├── static/
│   ├── css/style.css        # Styles (mode clair/sombre)
│   └── js/main.js           # Logique frontend
├── templates/
│   └── index.html           # Interface utilisateur
├── utils/
│   ├── audio_processing.py  # Traitement audio
│   └── file_handler.py      # Gestion des fichiers
├── datasets_scripts/        # Scripts d'entraînement
│   ├── prepare_wav2vec2_dataset.py   # Préparation dataset VoxPopuli
│   ├── finetune_wav2vec2.py          # Fine-tuning Wav2Vec2
│   ├── evaluate_language_models.py   # Évaluation des modèles
│   ├── train_language_classifier.py  # Entraînement CNN
│   ├── train_language_classifier_improved.py  # CNN avec augmentation
│   └── prepare_common_voice_data.py  # Préparation Common Voice
├── trained_models/          # Modèles entraînés
├── uploads/                 # Fichiers uploadés (temporaire)
├── config.py                # Configuration globale
└── requirements.txt         # Dépendances Python
```

---

## Auteur

- **Jean-Charles da Silva** - [@jcharlesDS](https://github.com/jcharlesDS)

---

## Notes importantes

- Les premiers téléchargements de modèles peuvent prendre plusieurs minutes
- Pour les fichiers longs (>30min), privilégiez le mode GPU
- La précision de la diarisation dépend de la qualité audio et du nombre de locuteurs

---
