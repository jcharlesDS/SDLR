import librosa
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict

class AudioLanguageClassifier(nn.Module):
    """Classificateur CNN pour la détection de la langue"""
    
    def __init__(self, n_mels=128, n_classes=2):
        super(AudioLanguageClassifier, self).__init__()
        
        # Architecture du CNN
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Adaptive pooling pour gérer les différentes longueurs d'entrée
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Couches entièrement connectées
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, n_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class LanguageClassifierInference:
    """Classe pour utiliser le modèle fine-tuné en production."""
    
    def __init__(self, model_path: str = "trained_models/lang_classifier.pth", device: str = "cpu"):
        self.device = device
        self.model = AudioLanguageClassifier()
        
        # Charger les poids fine-tunés
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Modèle fine-tuné chargé depuis {model_path}")
        else:
            print(f"ATTENTION: Modèle non trouvé à {model_path}, utilisation du modèle non entraîné")
        
        self.model.to(device)
        self.model.eval()
        
        self.label_map = {0: "fr", 1: "en"}
    
    def extract_melspectrogram(self, audio_path: str, duration: float = 10.0, max_time_frames: int = 300) -> np.ndarray:
        """Extrait un spectrogramme Mel d'un fichier audio."""
        # Charger l'audio et limiter à une durée maximale pour éviter les problèmes de mémoire
        y, sr = librosa.load(audio_path, sr=16000, duration=duration)
        
        # Spectrogramme Mel
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512
        )
        
        # Convertir en décibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normaliser
        mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
        
        # Padding ou truncate pour avoir une taille fixe (n_mels x max_time_frames)
        current_time = mel_spec_normalized.shape[1]
        if current_time < max_time_frames:
            # Padding à droite avec des zéros
            pad_width = max_time_frames - current_time
            mel_spec_normalized = np.pad(mel_spec_normalized, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate si trop long
            mel_spec_normalized = mel_spec_normalized[:, :max_time_frames]
        
        return mel_spec_normalized
    
    def predict(self, audio_path: str) -> Dict:
        """Prédit la langue d'un fichier audio."""
        # Extraire les features
        mel_spec = self.extract_melspectrogram(audio_path)
        
        # Convertir en tensor (batch=1, channel=1, height, width)
        x = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        language = self.label_map[predicted_class]
        
        return {
            "language": language,
            "confidence": confidence,
            "probabilities": {
                "fr": probabilities[0][0].item(),
                "en": probabilities[0][1].item()
            }
        }