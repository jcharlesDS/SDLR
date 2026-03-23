import librosa
import sys
import time
from pathlib import Path
from pydub import AudioSegment
from typing import Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))
from config import AUDIO_CONFIG, UPLOAD_DIR

class AudioFileHandler:
    """Classe pour gérer les opérations sur les fichiers audio."""
    
    def __init__(self):
        self.sample_rate = AUDIO_CONFIG["sample_rate"]
        self.max_duration = AUDIO_CONFIG["max_duration"]
        self.allowed_formats = AUDIO_CONFIG["allowed_formats"]
    
    def validate_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Valide un fichier audio
        
        Args:
            file_path (Path): Chemin du fichier à valider.
        
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Vérifier que le fichier existe
        if not file_path.exists():
            return False, "Le fichier n'existe pas."
        
        # Vérifier le format du fichier
        if file_path.suffix.lower() not in self.allowed_formats:
            return False, f"Format de fichier non supporté. Formats acceptés: {self.allowed_formats}"
        
        # Vérifier la taille du fichier
        file_size = file_path.stat().st_size
        if file_size > AUDIO_CONFIG["max_file_size"]:
            max_mb = AUDIO_CONFIG["max_file_size"] / (1024 * 1024)
            return False, f"Le fichier est trop volumineux. Taille maximale: {max_mb:.2f} MB."
        
        try:
            # Charger le fichier pour vérifier qu'il est valide
            duration = librosa.get_duration(path=str(file_path))
            
            # Vérifier la durée du fichier
            if duration > self.max_duration:
                return False, f"Le fichier est trop long. Durée maximale: (max {self.max_duration} secondes)."
            
            return True, None
        except Exception as e:
            return False, f"Erreur lors de la validation du fichier: {str(e)}"
    
    def convert_to_wav(self, input_path: Path, output_path: Optional[Path] = None) -> Path:
        """Convertit un fichier audio en format WAV (si nécessaire)
        
        Args:
            input_path (Path): Chemin du fichier d'entrée.
            output_path (Optional[Path]): Chemin du fichier de sortie. 
            Si None, le fichier sera enregistré dans UPLOAD_DIR avec le même nom mais extension .wav.
        
        Returns:
            Path: Chemin du fichier WAV généré.
        """
        
        # Si déjà en WAV, pas de conversion
        if input_path.suffix.lower() == ".wav":
            return input_path
        
        # Définir le chemin de sortie
        if output_path is None:
            output_path = input_path.with_suffix(".wav")
        
        try:
            # Charger avec pydub
            audio = AudioSegment.from_file(str(input_path))
            
            # Exporter en WAV
            audio = audio.set_channels(1)  # Convertir en mono
            audio = audio.set_frame_rate(self.sample_rate)  # Changer la fréquence d'échantillonnage
            audio.export(str(output_path), format="wav")
            
            return output_path
        
        except Exception as e:
            raise ValueError(f"Erreur lors de la conversion du fichier: {str(e)}")
        
    def get_audio_info(self, file_path: Path) -> dict:
        """Récupère les informations d'un fichier audio
        
        Args:
            file_path (Path): Chemin du fichier audio.
        
        Returns:
            dict: Dictionnaire contenant les informations du fichier audio.
        """
        try:
            # Changer l'audio
            y, sr = librosa.load(str(file_path), sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            return {
                "duration": round(duration, 2),
                "sample_rate": sr,
                "num_samples": len(y),
                "format": file_path.suffix.lower(),
                "file_size": file_path.stat().st_size,
                "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2)
            }
        
        except Exception as e:
            raise ValueError(f"Erreur lors de la récupération des informations du fichier: {str(e)}")
        
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """Nettoie les fichiers uploadés trop anciens
        
        Args:
            max_age_hours (int): Âge maximum des fichiers en heures.
            Les fichiers plus anciens seront supprimés.
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        print(f"Fichier supprimé: {file_path.name}")
                    except Exception as e:
                        print(f"Erreur lors de la suppression du fichier {file_path.name}: {str(e)}")