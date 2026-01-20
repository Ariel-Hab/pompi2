"""
api.py
Servidor con monitoreo y reinicio de RunPod integrado
"""
import sys
import httpx # Asegúrate de tenerlo instalado: pip install httpx
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.config import RUNPOD_CONFIG

# Configurar el path para importar tus módulos
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from chat_domain.manager import ConversationManager

# 1. Definir el modelo de datos
class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: str

class ResetRequest(BaseModel):
    session_id: str
    user_id: str

# 2. Inicializar la App
app = FastAPI(title="API Chat Veterinario")

# 3. Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SERVICIOS DE RUNPOD ---

@app.get("/api/runpod-status")
async def get_runpod_status():
    """Verifica si el pod está encendido o apagado."""
    url = f"https://api.runpod.io/graphql?api_key={RUNPOD_CONFIG['api_key']}"
    
    query = """
    query {
      pod(input: {podId: "%s"}) {
        runtime {
          status
        }
      }
    }
    """ % RUNPOD_CONFIG['endpoint_id']

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={'query': query}, timeout=10.0)
            
        if response.status_code == 200:
            data = response.json()
            status = data.get("data", {}).get("pod", {}).get("runtime", {}).get("status", "unknown")
            return {
                "pod_id": RUNPOD_CONFIG['endpoint_id'],
                "status": status,
                "is_ready": status == "running"
            }
        raise HTTPException(status_code=response.status_code, detail="Error en RunPod API")
    except Exception as e:
        return {"status": "error", "message": str(e), "is_ready": False}

@app.post("/api/runpod-resume")
async def resume_runpod():
    """Enciende el pod si está detenido."""
    url = f"https://api.runpod.io/graphql?api_key={RUNPOD_CONFIG['api_key']}"
    
    # Mutación para encender el pod (podResume)
    query = """
    mutation {
      podResume(input: {podId: "%s"}) {
        id
        desiredStatus
      }
    }
    """ % RUNPOD_CONFIG['endpoint_id']

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={'query': query}, timeout=15.0)
            
        if response.status_code == 200:
            return {"message": "Orden de encendido enviada", "data": response.json()}
        
        raise HTTPException(status_code=response.status_code, detail="No se pudo reiniciar el pod")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINTS DE CHAT ---

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        print(f"📩 Recibido: {request.message} | User: {request.user_id} | Sesión: {request.session_id}")
        
        # Pasamos el user_id al inicializar el Manager
        manager = ConversationManager(
            session_id=request.session_id, 
            user_id=request.user_id
        )
        
        # Ya no pasamos classification aquí
        response_payload = manager.handle_message(user_text=request.message)
        
        return response_payload
    except Exception as e:
        print(f"❌ Error en API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset-context")
async def reset_context_endpoint(request: ResetRequest):
    """Limpia la memoria de la conversación actual."""
    try:
        manager = ConversationManager(
            session_id=request.session_id, 
            user_id=request.user_id
        )
        
        success = manager.reset_session()
        
        if success:
            return {"status": "ok", "message": "Contexto reiniciado correctamente"}
        else:
            raise HTTPException(status_code=500, detail="No se pudo limpiar el historial")
            
    except Exception as e:
        print(f"❌ Error en API Reset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

