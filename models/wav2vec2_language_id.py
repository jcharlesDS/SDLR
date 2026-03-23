import librosa
import torch
from pathlib import Path
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor


class Wav2Vec2LanguageID:
    """Classificateur de langue basé sur Wav2Vec2 fine-tuné pour FR/EN
    
    Remplace le CNN fine-tuné précédent avec un modèle Wav2Vec2.
    Interface identique pour compatibilité avec le système de vote.
    """
    
    def __init__(self, model_path: str = "trained_models/wav2vec2-language-id", device: str = None):
        """
        Initialise le classificateur Wav2Vec2
        
        Args:
            model_path: Chemin vers le modèle Wav2Vec2 fine-tuné
            device: Device à utiliser ("cuda", "cpu", ou None pour auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Vérifier que le modèle fine-tuné existe
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Modèle Wav2Vec2 introuvable à {model_path}\n"
                f"   Veuillez d'abord fine-tuner le modèle avec les scripts d'entraînement."
            )
        
        print(f"Chargement du modèle Wav2Vec2 fine-tuné depuis {model_path}...")
        
        # Charger le modèle fine-tuné
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Modèle Wav2Vec2 chargé sur {self.device}")
        
        self.label2lang = {0: "en", 1: "fr"}  # Mapping IDs vers langues
    
    def predict(self, audio_path: str) -> dict:
        """
        Prédire la langue d'un fichier audio
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            dict: {"language": "fr/en", "confidence": float, "probabilities": {...}}
                Format identique au CNN pour compatibilité avec le système de vote
        """
        # Charger et pré-traiter l'audio (16kHz requis pour Wav2Vec2)
        speech, sr = librosa.load(audio_path, sr=16000)
        
        # Limiter la durée pour éviter les OOM (30 secondes max)
        max_samples = 16000 * 30  # 30 secondes
        if len(speech) > max_samples:
            speech = speech[:max_samples]
        
        # Extraire les features avec le feature extractor
        inputs = self.feature_extractor(
            speech, 
            sampling_rate=16000, 
            return_tensors="pt",
            padding=True
        )
        
        # Prédiction
        with torch.no_grad():
            input_values = inputs.input_values.to(self.device)
            logits = self.model(input_values).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_id = torch.argmax(probs, dim=-1).item()
        
        language = self.label2lang[predicted_id]
        confidence = probs[0][predicted_id].item()
        
        return {
            "language": language,
            "confidence": confidence,
            "probabilities": {
                "en": probs[0][0].item(),
                "fr": probs[0][1].item()
            }
        }