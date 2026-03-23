import json
import sys
import time
import uuid
import warnings
from app.models import AnalysisRequest, UploadResponse
from config import AUDIO_CONFIG, UPLOAD_DIR, LANGUAGE_NAMES, MODEL_CONFIG
from fastapi import APIRouter, File, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from io import BytesIO
from pathlib import Path
from pydub import AudioSegment

# Supprimer les warnings non critiques de pyannote/speechbrain
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*torchaudio backend.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*speechbrain.pretrained.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom.*")

# Détour pour corriger l'incompatibilité torchaudio 2.x avec speechbrain/pyannote
# IMPORTANT: Doit être fait AVANT d'importer SpeakerDiarizer
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: "soundfile"
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda backend: None

from models.language_id import LanguageIdentifier
from models.diarization import SpeakerDiarizer


sys.path.append(str(Path(__file__).parent.parent.parent))

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    """Upload un fichier audio"""
    # Vérifier l'extension du fichier
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in AUDIO_CONFIG["allowed_formats"]:
        raise HTTPException(status_code=400, detail=f"Format de fichier non supporté. Formats acceptés: {AUDIO_CONFIG['allowed_formats']}")
    
    # Générer un ID unique
    file_id = str(uuid.uuid4())
    
    # Lire le contenu du fichier
    content = await file.read()
    file_size = len(content)
    
    # Vérifier la taille du fichier
    if file_size > AUDIO_CONFIG["max_file_size"]:
        raise HTTPException(status_code=400, detail=f"Taille du fichier dépassée. Taille maximale: {AUDIO_CONFIG['max_file_size'] // (1024 * 1024)} Mo")
    
    # Sauvegarder le fichier
    save_path = UPLOAD_DIR / f"{file_id}{file_ext}"
    with open(save_path, "wb") as f:
        f.write(content)

    return UploadResponse(
        success=True,
        message="Fichier uploadé avec succès.",
        file_id=file_id,
        filename=file.filename,
        file_size=file_size,
        audio_url=f"/api/audio/stream/{file_id}{file_ext}"
    )

@router.get("/audio/stream/{filename}")
async def stream_audio(filename: str, request: Request):
    """Stream un fichier audio avec support du Range header pour la lecture en continu"""
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Fichier audio non trouvé")
    
    # Obtenir la taille du fichier
    file_size = file_path.stat().st_size
    
    # Obtenir l'en-tête Range de la requête
    range_header = request.headers.get("range")
    
    # Déterminer le type de contenu en fonction de l'extension
    ext = file_path.suffix.lower()
    content_type_map = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg"
    }
    content_type = content_type_map.get(ext, "audio/mpeg")
    
    # Si aucune plage n'est demandée, envoyer le fichier entier
    if not range_header:
        with open(file_path, "rb") as f:
            content = f.read()
        return Response(
            content=content,
            media_type=content_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size)
            }
        )
    
    # Parser l'en-tête Range
    try:
        range_value = range_header.replace("bytes=", "")
        range_parts = range_value.split("-")
        start = int(range_parts[0]) if range_parts[0] else 0
        end = int(range_parts[1]) if len(range_parts) > 1 and range_parts[1] else file_size - 1
    except (ValueError, IndexError):
        raise HTTPException(status_code=416, detail="En-tête Range invalide")
    
    # Valider la plage
    if start >= file_size or end >= file_size or start > end:
        raise HTTPException(status_code=416, detail="Plage non satisfaisable")
    
    # Lire le chunk demandé
    chunk_size = end - start + 1
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(chunk_size)
    
    # Retourner 206 Partial Content
    return Response(
        content=chunk,
        status_code=206,
        media_type=content_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size)
        }
    )

@router.post("/analyze")
async def analyze_audio(request: AnalysisRequest):
    """
    Analyser un fichier audio (langue, diarization + transcription optionnelle)
    """
    
    # Capturer le temps de début pour cette analyse spécifique
    analysis_start_time = time.time()
    
    try:
        # Chercher le fichier avec l'ID fourni
        matching_files = list(UPLOAD_DIR.glob(f"{request.file_id}.*"))
        if not matching_files:
            raise HTTPException(status_code=404, detail="Fichier non trouvé")
        
        file_path = str(matching_files[0])
        filename = matching_files[0].name
        
        # Déterminer le device basé sur le choix de l'utilisateur
        device = "cuda" if request.use_cuda else "cpu"
        print(f"Analyse demandée avec device: {device}")
        
        # Phase 1 : Détection de la langue
        print("Phase 1 : Détection de la langue...")
        lang_identifier = LanguageIdentifier(device=device)
        lang_result = lang_identifier.detect_language(file_path)
        
        # Phase 2 : Diarization (séparation des locuteurs)
        num_speakers_info = f", num_speakers={request.num_speakers}" if request.num_speakers else ""
        print(f"Phase 2 : Diarization (merge_collar={request.merge_collar}s{num_speakers_info})...")
        diarizer = SpeakerDiarizer(device=device)
        
        # Passer num_speakers si fourni par l'utilisateur
        if request.num_speakers:
            diarization_result = diarizer.diarize(
                file_path, 
                min_speakers=request.num_speakers,
                max_speakers=request.num_speakers,
                merge_collar=request.merge_collar
            )
        else:
            diarization_result = diarizer.diarize(file_path, merge_collar=request.merge_collar)
        
        # Phase 3 : Transcription (optionnelle)
        transcription_result = None
        if request.enable_transcription:
            print("Phase 3: Transcription...")
            from models.transcription import Transcriber
            transcriber = Transcriber(device=device, model_name=MODEL_CONFIG["whisper"]["model_size"])
            transcription_result = transcriber.transcribe_with_speakers(
                file_path,
                diarization_result["speakers"],
                language=lang_result["language"]
            )
            print(f"Transcription terminée: {len(transcription_result)} segments transcrits")
        
        # Calculer le temps de traitement pour cette analyse
        processing_time = round(time.time() - analysis_start_time, 2)
        
        return JSONResponse(
            content={
                "status": "success",
                "language": {
                    "language_code": lang_result["language"],
                    "language_name": LANGUAGE_NAMES.get(lang_result["language"], lang_result["language"]),
                    "confidence": lang_result["confidence"]
                },
                "num_speakers": diarization_result["num_speakers"],
                "speakers": diarization_result["speakers"],
                "transcription": transcription_result,
                "processing_time": processing_time
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Erreur lors de l'analyse : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse : {str(e)}")

@router.get("/analyze/stream")
async def analyze_audio_stream(
    file_id: str,
    enable_transcription: bool = False,
    use_cuda: bool = False,
    merge_collar: float = 1.0,
    num_speakers: int = None
):
    """
    Analyser un fichier audio avec progression en temps réel via Server-Sent Events
    """
    
    # Capturer le timestamp au niveau de la requête pour garantir l'isolation
    request_start_time = time.time()
    
    async def generate_progress():
        # Utiliser le timestamp capturé au niveau de la requête
        analysis_start = request_start_time
        
        try:
            # Envoyer un événement de début
            data = {'progress': 0, 'message': 'Initialisation...', 'step': 'init'}
            yield f"data: {json.dumps(data)}\n\n"
            
            # Chercher le fichier avec l'ID fourni
            matching_files = list(UPLOAD_DIR.glob(f"{file_id}.*"))
            if not matching_files:
                data = {'error': 'Fichier non trouvé'}
                yield f"data: {json.dumps(data)}\n\n"
                return
            
            file_path = str(matching_files[0])
            filename = matching_files[0].name
            
            # Déterminer le device
            device = "cuda" if use_cuda else "cpu"
            print(f"Analyse demandée avec device: {device}")
            
            # Phase 1 : Détection de la langue (0% -> 30%)
            data = {'progress': 5, 'message': 'Détection de la langue...', 'step': 'language'}
            yield f"data: {json.dumps(data)}\n\n"
            print("Phase 1 : Détection de la langue...")
            lang_identifier = LanguageIdentifier(device=device)
            lang_result = lang_identifier.detect_language(file_path)
            
            lang_name = LANGUAGE_NAMES.get(lang_result["language"], lang_result["language"])
            data = {'progress': 30, 'message': f'Langue détectée: {lang_name}', 'step': 'language_done'}
            yield f"data: {json.dumps(data)}\n\n"
            
            # Phase 2 : Diarization (30% -> 60%)
            data = {'progress': 35, 'message': 'Séparation des locuteurs...', 'step': 'diarization'}
            yield f"data: {json.dumps(data)}\n\n"
            num_speakers_info = f", num_speakers={num_speakers}" if num_speakers else ""
            print(f"Phase 2 : Diarization (merge_collar={merge_collar}s{num_speakers_info})...")
            diarizer = SpeakerDiarizer(device=device)
            
            # Passer num_speakers si fourni par l'utilisateur
            if num_speakers:
                diarization_result = diarizer.diarize(
                    file_path,
                    min_speakers=num_speakers,
                    max_speakers=num_speakers,
                    merge_collar=merge_collar
                )
            else:
                diarization_result = diarizer.diarize(file_path, merge_collar=merge_collar)
            
            data = {'progress': 60, 'message': f'{diarization_result["num_speakers"]} locuteur(s) détecté(s)', 'step': 'diarization_done'}
            yield f"data: {json.dumps(data)}\n\n"
            
            # Phase 3 : Transcription (60% -> 95%) si activée
            transcription_result = None
            if enable_transcription:
                data = {'progress': 65, 'message': 'Transcription en cours...', 'step': 'transcription'}
                yield f"data: {json.dumps(data)}\n\n"
                print("Phase 3: Transcription...")
                from models.transcription import Transcriber
                transcriber = Transcriber(device=device, model_name=MODEL_CONFIG["whisper"]["model_size"])
                transcription_result = transcriber.transcribe_with_speakers(
                    file_path,
                    diarization_result["speakers"],
                    language=lang_result["language"]
                )
                print(f"Transcription terminée: {len(transcription_result)} segments transcrits")
                data = {'progress': 95, 'message': 'Transcription terminée', 'step': 'transcription_done'}
                yield f"data: {json.dumps(data)}\n\n"
            else:
                data = {'progress': 95, 'message': 'Finalisation...', 'step': 'finalizing'}
                yield f"data: {json.dumps(data)}\n\n"
            
            # Finalisation (95% -> 100%)
            processing_time = round(time.time() - analysis_start, 2)
            
            # Envoyer le résultat final
            result = {
                "progress": 100,
                "message": "Analyse terminée",
                "step": "complete",
                "results": {
                    "status": "success",
                    "language": {
                        "language_code": lang_result["language"],
                        "language_name": LANGUAGE_NAMES.get(lang_result["language"], lang_result["language"]),
                        "confidence": lang_result["confidence"]
                    },
                    "num_speakers": diarization_result["num_speakers"],
                    "speakers": diarization_result["speakers"],
                    "transcription": transcription_result,
                    "processing_time": processing_time,
                    "device": device
                }
            }
            yield f"data: {json.dumps(result)}\n\n"
            
        except Exception as e:
            print(f"Erreur lors de l'analyse : {str(e)}")
            error_data = {'error': str(e), 'step': 'error'}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.delete("/uploads/clear")
async def clear_uploads():
    """Supprime tous les fichiers du dossier uploads"""
    try:
        import shutil
        deleted_count = 0
        
        if UPLOAD_DIR.exists():
            for file in UPLOAD_DIR.iterdir():
                if file.is_file():
                    file.unlink()
                    deleted_count += 1
        
        return JSONResponse(
            content={
                "status": "success",
                "message": f"{deleted_count} fichier(s) supprimé(s)"
            }
        )
    except Exception as e:
        print(f"Erreur lors du nettoyage : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du nettoyage : {str(e)}")

@router.get("/config/huggingface-token/status")
async def check_hf_token_status():
    """Vérifie si le token HuggingFace est configuré (sans le renvoyer)"""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    token = os.getenv("HUGGINGFACE_TOKEN")
    is_configured = token is not None and len(token.strip()) > 0
    
    return JSONResponse(
        content={
            "configured": is_configured
        }
    )

@router.post("/config/huggingface-token")
async def save_hf_token(request: dict):
    """Sauvegarde le token HuggingFace dans le fichier .env"""
    token = request.get("token", "").strip()
    
    if not token:
        raise HTTPException(status_code=400, detail="Token vide")
    
    # Vérifier que le token a un format valide (commence par hf_)
    if not token.startswith("hf_"):
        raise HTTPException(status_code=400, detail="Token invalide (doit commencer par 'hf_')")
    
    try:
        from pathlib import Path
        import os
        from dotenv import load_dotenv
        
        env_path = Path(__file__).parent.parent.parent / ".env"
        
        # Lire le contenu existant du .env
        lines = []
        token_found = False
        
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('HUGGINGFACE_TOKEN='):
                        lines.append(f'HUGGINGFACE_TOKEN={token}\n')
                        token_found = True
                    else:
                        lines.append(line)
        
        # Si le token n'était pas dans le fichier, l'ajouter
        if not token_found:
            lines.append(f'HUGGINGFACE_TOKEN={token}\n')
        
        # Écrire le fichier .env
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        # Recharger les variables d'environnement
        load_dotenv(override=True)
        os.environ['HUGGINGFACE_TOKEN'] = token
        
        # Mettre à jour la config en mémoire
        from config import MODEL_CONFIG
        MODEL_CONFIG["pyannote"]["auth_token"] = token
        
        print("✓ Token HuggingFace configuré avec succès")
        
        return JSONResponse(
            content={
                "status": "success",
                "message": "Token HuggingFace configuré avec succès"
            }
        )
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du token : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la sauvegarde : {str(e)}")

@router.get("/audio/extract-segment")
async def extract_audio_segment(
    file_id: str,
    start_time: float,
    end_time: float,
    format: str = "mp3"
):
    """
    Extrait un segment d'un fichier audio et le retourne pour téléchargement.
    
    Args:
        file_id: ID du fichier audio
        start_time: Début du segment en secondes
        end_time: Fin du segment en secondes
        format: Format de sortie (mp3 ou wav)
    """
    # Validation du format
    if format not in ["mp3", "wav"]:
        raise HTTPException(status_code=400, detail="Format non supporté. Utilisez 'mp3' ou 'wav'.")
    
    # Trouver le fichier source
    audio_files = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not audio_files:
        raise HTTPException(status_code=404, detail="Fichier audio non trouvé")
    
    source_file = audio_files[0]
    
    try:
        # Charger l'audio avec pydub
        audio = AudioSegment.from_file(str(source_file))
        
        # Convertir les temps en millisecondes
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        # Valider les temps
        if start_ms < 0 or end_ms > len(audio):
            raise HTTPException(status_code=400, detail="Plage de temps invalide")
        if start_ms >= end_ms:
            raise HTTPException(status_code=400, detail="Le temps de début doit être inférieur au temps de fin")
        
        # Extraire le segment
        segment = audio[start_ms:end_ms]
        
        # Exporter dans un buffer en mémoire
        buffer = BytesIO()
        
        if format == "mp3":
            segment.export(buffer, format="mp3", bitrate="192k")
            media_type = "audio/mpeg"
            file_extension = "mp3"
        else:  # wav
            segment.export(buffer, format="wav")
            media_type = "audio/wav"
            file_extension = "wav"
        
        # Remettre le curseur au début du buffer
        buffer.seek(0)
        
        # Générer un nom de fichier descriptif
        filename = f"segment_{int(start_time)}s-{int(end_time)}s.{file_extension}"
        
        # Retourner le fichier
        return Response(
            content=buffer.getvalue(),
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
        
    except Exception as e:
        print(f"Erreur lors de l'extraction du segment : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'extraction : {str(e)}") 