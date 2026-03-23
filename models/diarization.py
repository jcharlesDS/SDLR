import sys
import torch
import librosa
from pathlib import Path
from typing import Dict, List, Optional
from pyannote.audio import Pipeline

sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_CONFIG

class SpeakerDiarizer:
    """Séparation des locuteurs avec pyannote.audio."""
    
    def __init__(self, device: Optional[str] = None, auth_token: Optional[str] = None):
        """Initialise le pipeline de diarisation
        
        Args:
            device : Device à utiliser ("cuda", "cpu", ou "auto").
            auth_token : Token d'authentification HuggingFace nécessaire pour pyannote.
        """
        device_config = device or MODEL_CONFIG["pyannote"]["device"]
        auth_token = auth_token or MODEL_CONFIG["pyannote"]["auth_token"]
        self.auth_token = auth_token
        
        # Gérer le device
        if device_config == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_config
            if device_config == "cuda" and not torch.cuda.is_available():
                print(f"CUDA demandé mais non disponible, basculement sur CPU")
                self.device = "cpu"
        
        print(f"Chargement du pipeline de diarization sur {self.device}...")
        
        # Charger le pipeline pyannote
        self.pipeline = Pipeline.from_pretrained(MODEL_CONFIG["pyannote"]["pipeline"], token=self.auth_token)
        
        # Déplacer sur le device approprié
        if self.device == "cuda":
            self.pipeline.to(torch.device("cuda"))
        
        print(f"Pipeline de diarization chargé avec succès")
    
    def _merge_adjacent_segments(self, speakers: List[Dict], collar: float = 0.5) -> List[Dict]:
        """Fusionne les segments adjacents du même locuteur.
        
        Args:
            speakers: Liste des segments de parole
            collar: Écart maximal (en secondes) entre deux segments pour les fusionner
        
        Returns:
            Liste des segments fusionnés
        """
        if not speakers:
            return speakers
        
        # Trier par temps de début
        sorted_speakers = sorted(speakers, key=lambda x: x["start_time"])
        
        merged = []
        current = sorted_speakers[0].copy()
        
        for next_segment in sorted_speakers[1:]:
            # Si même locuteur et gap < collar, fusionner
            if (current["speaker_id"] == next_segment["speaker_id"] and 
                next_segment["start_time"] - current["end_time"] <= collar):
                # Étendre le segment courant
                current["end_time"] = next_segment["end_time"]
                current["duration"] = round(current["end_time"] - current["start_time"], 2)
            else:
                # Sauvegarder le segment courant et commencer un nouveau
                merged.append(current)
                current = next_segment.copy()
        
        # Ajouter le dernier segment
        merged.append(current)
        
        return merged
        
    def diarize(self, audio_path: str, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None, 
                merge_collar: float = 1.0) -> Dict:
        """Effectue la diarization d'un fichier audio
        
        Args:
            audio_path (str): Chemin vers le fichier audio à analyser.
            min_speakers (Optional[int]): Nombre minimum de locuteurs à détecter.
            max_speakers (Optional[int]): Nombre maximum de locuteurs à détecter.
            merge_collar (float): Écart maximal (en secondes) pour fusionner les segments adjacents du même locuteur.
            Valeur de 0 = pas de fusion, 0.5 = fusion si gap < 0.5s (ajuste selon les besoins)
        
        Returns:
            Dict: Dictionnaire contenant les segments de parole et les identifiants de locuteurs.
        """
        print("Diarization en cours...")
        
        # Charger l'audio avec librosa pour éviter les problèmes de compatibilité torchaudio
        print(f"Chargement de l'audio: {audio_path}")
        waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        
        # Convertir en tensor PyTorch et ajouter la dimension de canal
        audio_tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, time)
        
        # Créer le dictionnaire d'audio pour pyannote
        audio_dict = {
            "waveform": audio_tensor,
            "sample_rate": sample_rate
        }
        
        # Paramètres de diarization
        diarization_params = {}
        if min_speakers is not None:
            diarization_params["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_params["max_speakers"] = max_speakers
        
        # Exécuter la diarization avec le dictionnaire audio
        print(f"Lancement de la diarization...")
        diarization = self.pipeline(audio_dict, **diarization_params)
        
        # Convertir les résultats en format exploitable
        speakers = []
        speaker_mapping = {}
        speaker_counter = 1
        
        # Dans pyannote.audio 4.x, le résultat est un DiarizeOutput
        # Il faut accéder à l'attribut speaker_diarization
        if hasattr(diarization, 'speaker_diarization'):
            annotation = diarization.speaker_diarization
        elif hasattr(diarization, 'annotation'):
            annotation = diarization.annotation
        else:
            annotation = diarization
        
        # Itérer sur les segments
        for segment, track, speaker in annotation.itertracks(yield_label=True):
            # Mapper les labels pyannote (SPEAKER_00, etc.) vers des IDs simples
            if speaker not in speaker_mapping:
                speaker_mapping[speaker] = speaker_counter
                speaker_counter += 1
            
            speakers.append({
                "speaker_id": speaker_mapping[speaker],
                "start_time": round(segment.start, 2),
                "end_time": round(segment.end, 2),
                "duration": round(segment.end - segment.start, 2)
            })
        
        # Fusionner les segments adjacents du même locuteur
        if merge_collar > 0:
            print(f"Fusion des segments adjacents (collar={merge_collar}s)...")
            original_count = len(speakers)
            speakers = self._merge_adjacent_segments(speakers, collar=merge_collar)
            print(f"Segments fusionnés: {original_count} -> {len(speakers)}")
        
        # Filtrer les segments de durée trop courte (< 0.3s)
        # Ces segments courts sont souvent du bruit, des rires, ou des chevauchements
        min_duration = 0.3
        speakers_filtered = [s for s in speakers if s["duration"] >= min_duration]
        if len(speakers_filtered) < len(speakers):
            print(f"Segments filtrés (durée < {min_duration}s): {len(speakers)} -> {len(speakers_filtered)}")
        speakers = speakers_filtered
        
        # Recompter les locuteurs réellement présents après filtrage
        num_speakers = len(set(s["speaker_id"] for s in speakers)) if speakers else 0
        print(f"Diarization terminée: {num_speakers} locuteur(s) détecté(s)")
        
        return {
            "num_speakers": num_speakers,
            "speakers": speakers,
            "speaker_mapping": speaker_mapping
        }
        
    def get_pipeline_info(self) -> Dict:
        """Retourne les informations sur le pipeline"""
        return {
            "pipeline": MODEL_CONFIG["pyannote"]["pipeline"],
            "device": self.device,
            "cuda_available": torch.cuda.is_available()
        }