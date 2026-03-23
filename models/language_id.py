import sys
import torch
import whisper
from models.language_classifier import LanguageClassifierInference
from pathlib import Path
from typing import Dict, Optional

sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_CONFIG, LANGUAGE_NAMES

class LanguageIdentifier:
    """Détection de la langue avec Whisper et CNN.
    
    Utilise un système de vote entre deux modèles :
    - Whisper : Modèle de reconnaissance vocale avec détection de langue intégrée
    - CNN : Modèle fine-tuné spécifiquement pour la classification FR/EN
    """
    
    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None, use_finetuned: bool = True):
        """Initialise les modèles de détection de langue
        
        Args:
            model_size : Taille du modèle Whisper à charger (tiny, base, small, medium, large).
                        Si None, utilise la configuration de MODEL_CONFIG.
            device : Device à utiliser ("cuda", "cpu", ou "auto").
                    Si None, utilise la configuration de MODEL_CONFIG.
            use_finetuned : Si True, utilise le CNN fine-tuné en complément de Whisper 
                        pour une détection plus robuste via un système de vote.
        """
        # Utiliser les valeurs de MODEL_CONFIG si non spécifiées
        self.model_size = model_size or MODEL_CONFIG["whisper"]["model_size"]
        device_config = device or MODEL_CONFIG["whisper"]["device"]
        
        # Gérer le device "auto" avec diagnostic détaillé
        if device_config == "auto":
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                print(f"GPU détecté : {gpu_name}")
                print(f"CUDA version : {torch.version.cuda}")
            else:
                self.device = "cpu"
                print(f"GPU non détecté, utilisation du CPU")
                print(f"Pour utiliser le GPU, installez PyTorch avec CUDA et assurez-vous d'avoir une carte compatible.")
                
        else:
            self.device = device_config
            if device_config == "cuda" and not torch.cuda.is_available():
                print(f"CUDA demandé mais non disponible, basculement sur CPU")
                self.device = "cpu"
            
        print(f"Chargement du modèle Whisper '{self.model_size}' sur {self.device}...")

        # Charger le modèle Whisper
        self.model = whisper.load_model(self.model_size, device=self.device)
        
        print(f"Modèle Whisper '{self.model_size}' chargé avec succès")
        
        # Charger CNN fine-tuné si demandé
        self.use_finetuned = use_finetuned
        if use_finetuned:
            try:
                self.finetuned_classifier = LanguageClassifierInference(
                    model_path="trained_models/lang_classifier.pth",
                    device=self.device
                )
            except Exception as e:
                print(f"Erreur de chargement CNN: {e}")
                self.use_finetuned = False
    
    def detect_language(self, audio_path: str) -> Dict:
        """Détecte la langue d'un fichier audio avec Whisper et CNN
        
        Utilise un système de vote entre Whisper et le CNN fine-tuné :
        - Si accord : moyenne des confidences
        - Si désaccord : Whisper prioritaire si confiance > 50%, sinon CNN
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            Dict avec language, confidence, method, et détails des deux modèles
        """
        
        # Résultat Whisper
        whisper_result = self._detect_with_whisper(audio_path)
        
        result = {
            "language": whisper_result["language"],
            "confidence": whisper_result["confidence"],
            "method": "whisper"
        }
        
        # Si le classifieur fine-tuné est disponible, l'utiliser pour une détection parallèle
        if self.use_finetuned:
            finetuned_result = self.finetuned_classifier.predict(audio_path)
            
            # Afficher les deux prédictions
            print(f"\nDétection de langue:")
            print(f"   Whisper:     {whisper_result['language'].upper()} (confiance: {whisper_result['confidence']:.1%})")
            print(f"   CNN:         {finetuned_result['language'].upper()} (confiance: {finetuned_result['confidence']:.1%})")
            
            # Combiner les prédictions avec une logique améliorée
            if whisper_result["language"] == finetuned_result["language"]:
                # Si les deux méthodes sont en accord, augmenter la confiance
                result["confidence"] = (whisper_result["confidence"] + finetuned_result["confidence"]) / 2
                result["method"] = "ensemble"
                print(f"Accord: {result['language'].upper()} (confiance moyenne: {result['confidence']:.1%})")
            else:
                # Si désaccord, privilégier Whisper tant qu'il est au-dessus de 50%
                if whisper_result["confidence"] >= 0.5:
                    result["method"] = "whisper-priority"
                    print(f"Désaccord: Whisper gardé ({result['language'].upper()}, confiance {whisper_result['confidence']:.1%})")
                else:
                    # Si Whisper est peu confiant (<50%), utiliser le CNN
                    result["language"] = finetuned_result["language"]
                    result["confidence"] = finetuned_result["confidence"]
                    result["method"] = "finetuned-override"
                    print(f"Désaccord: CNN utilisé car Whisper peu confiant ({result['language'].upper()}, confiance Whisper {whisper_result['confidence']:.1%})")
                    
            result["whisper"] = whisper_result
            result["finetuned"] = finetuned_result
        
        return result

    def _detect_with_whisper(self, audio_path: Path) -> Dict[str, any]:
        """Détecte la langue d'un fichier audio via Whisper
        
        Args:
            audio_path (Path): Chemin vers le fichier audio à analyser.
        
        Returns:
            Dict[str, any]: Dictionnaire contenant les résultats de la détection de langue.
            ID de la langue, nom de la langue, et score de confidence.
        """
        try:
            # Charger et décoder l'audio
            audio = whisper.load_audio(str(audio_path))
            audio = whisper.pad_or_trim(audio)
            
            # Créer le spectrogramme Mel
            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            
            # Détecter la langue
            _, probs = self.model.detect_language(mel)
            
            # Récupérer la langue avec la plus haute probabilité
            detected_lang = max(probs, key=probs.get)
            confidence = probs[detected_lang]
            
            # Utiliser les noms de langues depuis la configuration
            result = {
                "language": detected_lang,
                "language_name": LANGUAGE_NAMES.get(detected_lang, detected_lang),
                "confidence": float(confidence),
                "all_probabilities": {k: float(v) for k, v in probs.items()}
            }
            
            return result

        except Exception as e:
            raise RuntimeError(f"Erreur lors de la détection de langue: {str(e)}")
        
    def detect_language_from_segments(self, audio_path: Path, max_segments: int = 3) -> Dict[str, any]:
        """Détecte la langue en découpant l'audio en segments (plus robuste)
        
        Args:
            audio_path (Path): Chemin vers le fichier audio à analyser.
            max_segments (int): Nombre maximum de segments à analyser pour la détection de langue.
        
        Returns:
            Dict[str, any]: Dictionnaire contenant les résultats de la détection de langue.
        """
        try:
            # Charger l'audio
            audio = whisper.load_audio(str(audio_path))
            
            # Découper en segments
            segment_length = len(audio) // max_segments
            language_votes = {}
            
            for i in range(max_segments):
                start = i * segment_length
                end = (i + 1) * segment_length
                segment = audio[start:end]
                
                # Pad ou trim pour Whisper
                segment = whisper.pad_or_trim(segment)
                
                # Spectrogramme Mel
                mel = whisper.log_mel_spectrogram(segment).to(self.device)
                
                # Détecter la langue
                _, probs = self.model.detect_language(mel)
                detected = max(probs, key=probs.get)
                
                # Compter les votes pour chaque langue
                language_votes[detected] = language_votes.get(detected, 0) + probs[detected]
                
            # Déterminer la langue majoritaire
            final_language = max(language_votes, key=language_votes.get)
            avg_confidence = language_votes[final_language] / max_segments
            
            return {
                "language": final_language,
                "language_name": LANGUAGE_NAMES.get(final_language, final_language),
                "confidence": float(avg_confidence),
                "votes": {k: float(v) for k, v in language_votes.items()}
            }
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la détection de langue par segments: {str(e)}")
        
    def get_model_info(self) -> Dict[str, any]:
        """Retourne des informations sur le modèle chargé"""
        
        return {
            "model_name": "Whisper",
            "model_size": self.model_size,
            "architecture": "Transformer (Encoder-Decoder)",
            "device": self.device,
            "parameters": {
                "tiny": "39M",
                "base": "74M",
                "small": "244M",
                "medium": "769M",
                "large": "1550M"
            }.get(self.model_size, "Unknown")
        }
