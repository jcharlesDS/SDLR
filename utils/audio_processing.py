import librosa
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict

sys.path.append(str(Path(__file__).parent.parent))
from config import AUDIO_CONFIG

class AudioProcessor:
    """Classe pour le prétraitement audio de base.
    
    Note : Whisper et pyannote font leur propre preprocessing interne.
    
    Cette classe fournit uniquement les opérations essentielles :
    - Chargement et normalisation audio
    - Suppression des silences
    - Segmentation audio
    - Extraction de métadonnées
    """
    
    def __init__(self):
        self.sample_rate = AUDIO_CONFIG["sample_rate"]
        
    def load_audio(self, file_path: Path, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Charge un fichier audio.
        
        Args:
            file_path (Path): Chemin vers le fichier audio.
            sr (Optional[int]): Taux d'échantillonnage. Si None, utilise le taux d'origine.
        
        Returns:
            Tuple[np.ndarray, int]: Tuple contenant le signal audio et le taux d'échantillonnage.
        """
        if sr is None:
            sr = self.sample_rate
        
        try:
            # Changer l'audio
            y, sr = librosa.load(str(file_path), sr=sr, mono=True)
            return y, sr
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement de l'audio: {str(e)}")
        
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalise le signal audio.
        (amplitude entre -1 et 1)
        
        Args:
            audio (np.ndarray): Signal audio à normaliser.
        
        Returns:
            np.ndarray: Signal audio normalisé.
        """
        # Normalisation par la valeur maximale absolue
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def remove_silence(self, audio: np.ndarray, top_db: int = 30) -> np.ndarray:
        """Supprime les silences du signal audio.
        
        Args:
            audio (np.ndarray): Signal audio à traiter.
            top_db (int): Seuil en dB pour considérer une partie comme silence.
        
        Returns:
            np.ndarray: Signal audio sans les silences.
        """
        # Detection des intervalles non silencieux
        intervals = librosa.effects.split(audio, top_db=top_db)
        
        if len(intervals) == 0:
            return audio  # Aucun silence détecté, retourner l'audio original
        
        # Concaténer les segments non silencieux
        trimmed = np.concatenate([audio[start:end] for start, end in intervals])
        return trimmed
    
    def segment_audio(self, audio: np.ndarray, sr: int, window_size: float, hop_size: float) -> list:
        """Découpe l'audio en segments (utile pour la diarization).
        
        Args:
            audio (np.ndarray): Signal audio à segmenter.
            sr (int): Taux d'échantillonnage du signal audio.
            window_size (float): Taille de la fenêtre en secondes.
            hop_size (float): Pas de déplacement entre les fenêtres en secondes.
        
        Returns:
            list: Liste des segments audio découpés.
        """
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)
        
        segments = []
        for start in range(0, len(audio) - window_samples + 1, hop_samples):
            end = start + window_samples
            segment = audio[start:end]
            segments.append({
                'audio': segment,
                'start_time': start / sr,
                'end_time': end / sr
            })
        
        return segments
    
    def process_file_for_whisper(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """
        Pipeline simplifié pour preprocessing avant Whisper
        (Whisper fera son propre preprocessing mel-spectrogram)
        
        Args:
            file_path: Chemin vers le fichier audio
            
        Returns:
            (audio_array, metadata)
        """
        # 1. Charger l'audio
        audio, sr = self.load_audio(file_path)
        
        # 2. Normaliser (optionnel, Whisper le fait aussi)
        audio = self.normalize_audio(audio)
        
        # 3. Métadonnées
        metadata = {
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'num_samples': len(audio)
        }
        
        return audio, metadata
    
    def get_audio_duration(self, file_path: Path) -> float:
        """
        Récupère uniquement la durée d'un fichier audio (utile pour UI)
        
        Args:
            file_path: Chemin vers le fichier audio
            
        Returns:
            Durée en secondes
        """
        audio, sr = self.load_audio(file_path)
        return len(audio) / sr
    