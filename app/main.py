import uvicorn
import sys
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Ajouter le parent directory au path
sys.path.append(str(Path(__file__).parent.parent))

from config import API_CONFIG, STATIC_DIR, BASE_DIR, UPLOAD_DIR
from app.routes import audio

# Initialiser l'application FastAPI
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"]
)

# Monter les dossiers statiques
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Templates Jinja2
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Inclure les routes de l'API
app.include_router(audio.router, prefix="/api", tags=["Audio Processing"])

# Route pour la page d'accueil
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Gestion des health checks
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API is running."}

# Point d'entrée pour lancer l'application
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=True
    )