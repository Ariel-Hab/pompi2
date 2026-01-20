import os
import time
import requests
import json
from dotenv import load_dotenv

# Cargar entorno
load_dotenv()

# --- CONFIGURACIÓN ---
API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

# RECUPERACIÓN INTELIGENTE DE LA URL
# Prioriza la variable de entorno, si no existe, intenta construir la de Serverless estándar
raw_url = os.getenv('RUNPOD_BASE_URL')
if not raw_url and ENDPOINT_ID:
    # URL Estándar de Serverless (no la de Proxy de Pods)
    raw_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/openai/v1"

# Limpieza de URL: Asegurar que termine en /v1 y no tenga espacios
if raw_url:
    BASE_URL = raw_url.strip().rstrip('/')
    if not BASE_URL.endswith('/v1'):
        BASE_URL += '/v1'
else:
    BASE_URL = "URL_NO_DEFINIDA"

def print_header(title):
    print(f"\n{'='*60}")
    print(f"📡 {title}")
    print(f"{'='*60}")

def print_status(step, status, msg, detail=None):
    icon = "✅" if status else "❌"
    print(f"{icon} [{step:<15}]: {msg}")
    if detail:
        print(f"   └── 📝 {detail}")

def monitor_runpod():
    print_header(f"DIAGNÓSTICO DE RUNPOD: {ENDPOINT_ID}")
    print(f"🔗 URL Base: {BASE_URL}")
    print(f"🔑 API Key:  {'*' * 6}{API_KEY[-4:] if API_KEY else 'NO DETECTADA'}")
    
    if BASE_URL == "URL_NO_DEFINIDA":
        print_status("CONFIG", False, "Falta RUNPOD_BASE_URL o RUNPOD_ENDPOINT_ID en .env")
        return

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # ---------------------------------------------------------
    # PASO 1: Listar Modelos (Verifica Auth + Persistencia)
    # ---------------------------------------------------------
    detected_model = None
    try:
        start_time = time.time()
        print(f"\n⏳ Consultando {BASE_URL}/models ...")
        
        response = requests.get(f"{BASE_URL}/models", headers=headers, timeout=30)
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            models_list = data.get('data', [])
            
            if not models_list:
                print_status("CONEXIÓN", True, f"Latencia: {latency:.0f}ms")
                print_status("MODELOS", False, "La lista de modelos está VACÍA.")
                print("   ⚠️  DIAGNÓSTICO: El contenedor responde, pero NO tiene modelos cargados.")
                print("   👉 SOLUCIÓN: Verifica el 'Network Volume' y la variable OLLAMA_MODEL.")
                return
            
            # Agarramos el primer modelo disponible
            detected_model = models_list[0]['id']
            print_status("CONEXIÓN", True, "Conexión exitosa")
            print_status("MODELOS", True, f"Modelo encontrado: '{detected_model}'", detail=f"Total disponibles: {len(models_list)}")
            
        else:
            print_status("CONEXIÓN", False, f"Error HTTP {response.status_code}")
            try:
                print(f"   🔎 Respuesta Raw: {response.json()}")
            except:
                print(f"   🔎 Respuesta Raw: {response.text}")
            
            if response.status_code == 401:
                print("   👉 SOLUCIÓN: Tu API Key es incorrecta o expiró.")
            if response.status_code == 404:
                print("   👉 SOLUCIÓN: La URL está mal formada. Verifica si sobra o falta '/v1'.")
            return

    except Exception as e:
        print_status("CONEXIÓN", False, "Excepción de Red")
        print(f"   ⚠️  Error: {e}")
        return

    # ---------------------------------------------------------
    # PASO 2: Test de Inferencia (Warm Up & Generación)
    # ---------------------------------------------------------
    print_header("TEST DE INFERENCIA (Warm Up)")
    
    # Si encontramos un modelo, usamos ese nombre. Si no, fallback a 'llama3'
    target_model = detected_model if detected_model else "llama3"
    
    payload = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": "Eres un test."},
            {"role": "user", "content": "Responde solo con la palabra: FUNCIONA"}
        ],
        "max_tokens": 10
    }

    try:
        print(f"🚀 Enviando request a modelo: '{target_model}'...")
        start_time = time.time()
        
        # Timeout alto (120s) porque si es Cold Start tarda mucho
        response = requests.post(
            f"{BASE_URL}/chat/completions", 
            headers=headers, 
            json=payload,
            timeout=120 
        )
        latency = (time.time() - start_time)
        
        if response.status_code == 200:
            content = response.json()
            reply = content['choices'][0]['message']['content']
            print_status("INFERENCIA", True, "Generación exitosa")
            print_status("RESPUESTA", True, f"'{reply.strip()}'", detail=f"Tardó {latency:.2f}s")
            
            if latency > 20:
                print("\n🐢 DIAGNÓSTICO: Cold Start detectado (tardó > 20s).")
                print("   El próximo request será mucho más rápido.")
            else:
                print("\n⚡ DIAGNÓSTICO: El sistema está caliente y rápido.")
                
        else:
            print_status("INFERENCIA", False, f"Falló con status {response.status_code}")
            try:
                err_json = response.json()
                print(f"   🔎 Detalles del servidor: {err_json}")
                if 'error' in err_json:
                    print(f"   ⚠️  Mensaje de error: {err_json['error']}")
            except:
                print(f"   🔎 Texto Raw: {response.text}")

    except requests.exceptions.ReadTimeout:
        print_status("INFERENCIA", False, "TIMEOUT (Se acabó el tiempo de espera)")
        print("   👉 DIAGNÓSTICO: El modelo está tardando demasiado en cargar (>120s).")
        print("   Posiblemente está descargando el modelo de internet (Checkea tu Volume).")
        
    except Exception as e:
        print_status("INFERENCIA", False, f"Error crítico: {e}")

if __name__ == "__main__":
    monitor_runpod()