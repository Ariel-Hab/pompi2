"""
Extracción de datos desde CSV (inicial) o API REST (incremental)
"""
from datetime import datetime
from typing import List, Dict, Optional
from core.config import ETL_CONFIG

# Importar parsers
from etl_domain.csv_parser import parse_categories_csv, parse_companies_csv, parse_offerproducts_csv, parse_products_csv, parse_offers_csv, parse_vademecum_csv
from etl_domain.api_client import ProductionAPIClient

def extract_companies_from_csv() -> List[Dict]:
    """
    Extrae empresas desde archivo CSV (carga inicial)
    """
    return parse_companies_csv()

def extract_offer_products_from_csv():
    """
    Lee la tabla intermedia que vincula Offers <-> Products.
    """
    return parse_offerproducts_csv()
def extract_vademecum_from_csv() -> List[Dict]:
    """
    Extrae registros del Vademécum Clínico desde archivo CSV (carga inicial).
    Delega la lógica al parser especializado.
    
    Returns:
        Lista de diccionarios con datos del Vademécum
    """
    return parse_vademecum_csv()

def extract_categories_from_csv():
    """Lee el archivo de categorías y devuelve una lista de dicts"""
    # Ajusta el nombre del archivo según corresponda
    return parse_categories_csv()

def extract_products_from_csv() -> List[Dict]:
    """
    Extrae productos desde archivo CSV (carga inicial)
    
    Returns:
        Lista de diccionarios con datos de productos
    """
    return parse_products_csv()


def extract_offers_from_csv() -> List[Dict]:
    """
    Extrae ofertas desde archivo CSV (carga inicial)
    
    Returns:
        Lista de ofertas
    """
    return parse_offers_csv()


def extract_products_from_api(last_sync: Optional[datetime] = None) -> List[Dict]:
    """
    Extrae productos desde API REST (sincronización incremental)
    
    Args:
        last_sync: Fecha del último sync
    
    Returns:
        Lista de productos actualizados
    """
    api_client = ProductionAPIClient()
    return api_client.get_products_updated_since(last_sync)


def extract_offers_from_api(last_sync: Optional[datetime] = None) -> List[Dict]:
    """
    Extrae ofertas desde API REST (sincronización incremental)
    
    Args:
        last_sync: Fecha del último sync
    
    Returns:
        Lista de ofertas actualizadas
    """
    api_client = ProductionAPIClient()
    return api_client.get_offers_updated_since(last_sync)


def get_last_sync_timestamp() -> Optional[datetime]:
    """
    Lee el archivo de última sincronización
    
    Returns:
        Datetime del último sync o None si no existe
    """
    try:
        sync_file = ETL_CONFIG['sync_file_path']
        with open(sync_file, 'r') as f:
            timestamp_str = f.read().strip()
            return datetime.fromisoformat(timestamp_str)
    except FileNotFoundError:
        print("ℹ️  Archivo de sync no encontrado (primera ejecución)")
        return None
    except Exception as e:
        print(f"⚠️  Error leyendo last_sync: {e}")
        return None


def save_last_sync_timestamp(timestamp: datetime):
    """
    Guarda el timestamp del último sync exitoso
    
    Args:
        timestamp: Datetime a guardar
    """
    try:
        sync_file = ETL_CONFIG['sync_file_path']
        with open(sync_file, 'w') as f:
            f.write(timestamp.isoformat())
        print(f"✅ Guardado last_sync: {timestamp}")
    except Exception as e:
        print(f"❌ Error guardando last_sync: {e}")