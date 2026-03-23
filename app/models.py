from datetime import datetime
from pydantic import BaseModel, Field, validator
from typing import Optional, List

# Modèle pour la réponse d'upload
class UploadResponse(BaseModel):
    success: bool
    message: str
    file_id: str
    filename: str
    file_size: int
    audio_url: str  # URL pour accéder au fichier audio

# Modèle pour les résultats de l'identification de langue
class LanguageResult(BaseModel):
    language : str # Code de la langue (ex: "fr", "en")
    confidence: float # Confiance du modèle pour cette langue
    language_name: str # Nom de la langue (ex: "Français", "Anglais")

# Modèle pour un segment de locuteur dans la diarization
class SpeakerSegment(BaseModel):
    speaker_id: int # ID du locuteur
    start_time: float # en secondes
    end_time: float # en secondes
    duration: float # en secondes
    
    @validator("duration", always=True)
    def calculate_duration(cls, v, values):
        if "start_time" in values and "end_time" in values:
            return round(values["end_time"] - values["start_time"], 2)
        return v

# Modèle pour la transcription (optionnelle)
class TranscriptionSegment(BaseModel):
    speaker_id: int # ID du locuteur
    start_time: float # en secondes
    end_time: float # en secondes
    text: str # Texte transcrit
    confidence: float # Confiance du modèle pour cette transcription

# Modèle pour les résultats complets de l'analyse audio
class AnalysisResult(BaseModel):
    file_id: str
    filename: str
    duration: float # Durée totale de l'audio en secondes
    language: LanguageResult
    speakers: List[SpeakerSegment]
    num_speakers: int
    transcription: Optional[List[TranscriptionSegment]] = None
    processing_time: float # Temps de traitement en secondes
    timestamp: datetime = Field(default_factory=datetime.now) # Timestamp de l'analyse

# Modèle pour la requête d'analyse audio
class AnalysisRequest(BaseModel):
    file_id: str
    enable_transcription: bool = False # Option pour activer la transcription
    use_cuda: bool = False # Option pour utiliser CUDA (GPU) au lieu du CPU
    merge_collar: float = 1.0 # Écart maximal (en secondes) pour fusionner les segments adjacents du même locuteur
    num_speakers: Optional[int] = None # Nombre de locuteurs attendu (optionnel, None = détection automatique)