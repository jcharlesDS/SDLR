import sys
import torch
import whisper
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_CONFIG

class Transcriber:
    """Transcription audio avec Whisper."""
    
    def __init__(self, device: Optional[str] = None, model_name: str = "small"):
        """Initialise le modèle de transcription Whisper
        
        Args:
            device : Device à utiliser ("cuda", "cpu", ou "auto").
            model_name : Taille du modèle Whisper à charger (tiny, base, small, medium, large).
        """
        device_config = device or MODEL_CONFIG["whisper"]["device"]
        
        if device_config == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_config
            
        print(f"Chargement du modèle de transcription Whisper '{model_name}' sur {self.device}...")
        self.model = whisper.load_model(model_name, device=self.device)
        print(f"Modèle de transcription Whisper '{model_name}' chargé avec succès")
        
    def transcribe(self, audio_path: str, language: str = None) -> Dict:
        """Transcrit un fichier audio
        
        Args:
            audio_path : Chemin vers le fichier audio à transcrire.
            language : Langue du fichier audio (optionnel).
        
        Returns:
            Dict : Résultat de la transcription.
        """
        print(f"Transcription en cours (langue : {language or 'détection automatique'})...")
        
        # Options de transcription
        options = {
            "task": "transcribe",
            "verbose": False
        }
        
        if language:
            options["language"] = language
        
        # Transcription
        result = self.model.transcribe(audio_path, **options)
        
        # Extraire les segments avec timestamps
        segments = []
        for seg in result["segments"]:
            segments.append({
                "start_time": round(seg["start"], 2),
                "end_time": round(seg["end"], 2),
                "text": seg["text"].strip(),
                "confidence": round(seg.get("no_speech_prob", 0.0), 3)
            })
        
        return {
            "text": result["text"].strip(),
            "segments": segments,
            "language": result.get("language", language)
        }
        
    def transcribe_with_speakers(self, audio_path: str, speaker_segments: List[Dict], language: str = None) -> List[Dict]:
        """Transcrit en alignant avec les segments de diarization
        
        Args:
            audio_path: Chemin vers le fichier audio
            speaker_segments: Liste des segments de diarization
            language: Code langue
        
        Returns:
            Liste de segments avec speaker_id et transcription
        """
        # D'abord transcrire tout l'audio
        transcription = self.transcribe(audio_path, language)
        
        # Aligner les segments de transcription avec les locuteurs
        result = []
        used_transcript_segments = set()  # Pour éviter les duplications
        
        for speaker_seg in speaker_segments:
            # Trouver les segments de transcription qui chevauchent ce segment de locuteur
            speaker_text = []
            
            for idx, trans_seg in enumerate(transcription["segments"]):
                # Ignorer si déjà utilisé
                if idx in used_transcript_segments:
                    continue
                
                # Vérifier si le segment de transcription chevauche le segment du locuteur
                overlap_start = max(speaker_seg["start_time"], trans_seg["start_time"])
                overlap_end = min(speaker_seg["end_time"], trans_seg["end_time"])
                overlap_duration = overlap_end - overlap_start
                
                if overlap_duration > 0:  # Il y a chevauchement
                    # Calculer le pourcentage de chevauchement
                    trans_duration = trans_seg["end_time"] - trans_seg["start_time"]
                    overlap_ratio = overlap_duration / trans_duration if trans_duration > 0 else 0
                    
                    # Seuil à 40% pour être plus tolérant (début/fin peuvent être décalés)
                    if overlap_ratio > 0.4:
                        speaker_text.append(trans_seg["text"])
                        used_transcript_segments.add(idx)  # Marquer comme utilisé
            
            # Combiner le texte pour ce locuteur
            combined_text = " ".join(speaker_text).strip()
            
            if combined_text:  # Ajouter seulement si du texte a été trouvé
                result.append({
                    "speaker_id": speaker_seg["speaker_id"],
                    "start_time": speaker_seg["start_time"],
                    "end_time": speaker_seg["end_time"],
                    "text": combined_text,
                    "confidence": 0.0  # À améliorer si besoin
                })
        
        # Gérer les segments de transcription orphelins (non assignés)
        # Ceci aide à récupérer les premières phrases qui peuvent être décalées temporellement
        orphan_segments = [
            (idx, seg) for idx, seg in enumerate(transcription["segments"]) 
            if idx not in used_transcript_segments
        ]
        
        if orphan_segments:
            print(f"{len(orphan_segments)} segment(s) orphelin(s), assignation...")
            
            first_speaker = min(speaker_segments, key=lambda s: s["start_time"]) if speaker_segments else None
            
            for idx, trans_seg in orphan_segments:
                best_speaker_id = None
                
                # Règle spéciale : segments au début (<1s) → premier locuteur
                if trans_seg["start_time"] < 1.0 and first_speaker:
                    best_speaker_id = first_speaker["speaker_id"]
                    print(f"   Début {trans_seg['start_time']:.1f}s → Locuteur {best_speaker_id}")
                else:
                    # Trouver le segment de locuteur le plus proche temporellement
                    min_distance = float('inf')
                    
                    for speaker_seg in speaker_segments:
                        # Distance entre les milieux des segments
                        trans_mid = (trans_seg["start_time"] + trans_seg["end_time"]) / 2
                        speaker_mid = (speaker_seg["start_time"] + speaker_seg["end_time"]) / 2
                        distance = abs(trans_mid - speaker_mid)
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_speaker_id = speaker_seg["speaker_id"]
                    
                    if min_distance <= 10.0:  # Augmenté à 10s pour les longs audios
                        print(f"   {trans_seg['start_time']:.1f}s → Locuteur {best_speaker_id} (dist: {min_distance:.1f}s)")
                
                # Créer un nouveau segment plutôt que fusionner (préserve l'ordre chronologique)
                if best_speaker_id is not None:
                    result.append({
                        "speaker_id": best_speaker_id,
                        "start_time": trans_seg["start_time"],
                        "end_time": trans_seg["end_time"],
                        "text": trans_seg["text"],
                        "confidence": 0.0
                    })
        
        # Trier les segments par temps pour affichage chronologique
        result.sort(key=lambda x: x["start_time"])
        
        return result
