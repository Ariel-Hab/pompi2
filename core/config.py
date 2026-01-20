"""
Configuración centralizada del sistema RAG
Carga variables de entorno y define constantes
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# ==================== API DE PRODUCCIÓN ====================
PRODUCTION_API_CONFIG = {
    'base_url': os.getenv('PRODUCTION_API_BASE_URL', 'https://www.integhra.com/api'),
    'token': os.getenv('PRODUCTION_API_TOKEN', ''),
    'products_endpoint': os.getenv('PRODUCTION_API_PRODUCTS', '/products/'),
    'offers_endpoint': os.getenv('PRODUCTION_API_OFFERS', '/offers/'),
    'timeout': int(os.getenv('API_TIMEOUT', 30)),
    'retry_attempts': int(os.getenv('API_RETRY_ATTEMPTS', 3)),
}

# ==================== PGVECTOR ====================
PGVECTOR_CONFIG = {
    'host': os.getenv('PGVECTOR_HOST', 'pgvector'),
    'port': int(os.getenv('PGVECTOR_PORT', 5432)),
    'database': os.getenv('PGVECTOR_DB_NAME', 'vectordb'),
    'user': os.getenv('PGVECTOR_USER', 'vectoruser'),
    'password': os.getenv('PGVECTOR_PASSWORD', 'vectorpass'),
}

# ==================== RUNPOD LLM ====================
RUNPOD_CONFIG = {
    'api_key': os.getenv('RUNPOD_API_KEY', ''),
    'endpoint_id': os.getenv('RUNPOD_ENDPOINT_ID', 'icdc5n1n38q0ke'),
    'base_url': os.getenv('RUNPOD_BASE_URL', 
                          f"https://{os.getenv('RUNPOD_ENDPOINT_ID')}-80.proxy.runpod.net/v1"),
}

# ==================== EMBEDDINGS ====================
EMBEDDING_CONFIG = {
    'model_name': os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
    'dimension': int(os.getenv('EMBEDDING_DIMENSION', 384)),
}

# ==================== ETL ====================
ETL_CONFIG = {
    'batch_size': int(os.getenv('BATCH_SIZE', 100)),
    'sync_file_path': os.getenv('SYNC_FILE_PATH', '/app/data/last_sync.txt'),
    'csv_products_path': os.getenv('CSV_PRODUCTS_PATH', '/app/data/products.csv'),
    'csv_offers_path': os.getenv('CSV_OFFERS_PATH', '/app/data/offers.csv'),
}

# ==================== PATHS ====================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

def get_pgvector_db_url():
    """Genera URL de conexión para pgvector"""
    cfg = PGVECTOR_CONFIG
    return f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"