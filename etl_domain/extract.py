"""
Extracción de datos desde CSV (inicial) o API REST (incremental)
VERSIÓN 4.0: Incluye productos.csv (base de datos REAL)
"""
from datetime import datetime
from typing import List, Dict, Optional
from core.config import ETL_CONFIG

# Importar parsers
from etl_domain.csv_parser import (
    parse_categories_csv,
    parse_companies_csv,
    parse_offerproducts_csv,
    parse_products_csv,      # ← CSV real (tu base)
    parse_offers_csv,
    parse_vademecum_csv      # ← CSV clínico (datos médicos)
)
from etl_domain.api_client import ProductionAPIClient


# ═══════════════════════════════════════════════════════════════
# EXTRACCIÓN DE PRODUCTOS REALES (productos.csv)
# ═══════════════════════════════════════════════════════════════

def extract_real_products_from_csv() -> List[Dict]:
    """
    Extrae productos desde productos.csv (TU BASE DE DATOS REAL)
    
    Este es tu catálogo comercial con:
    - IDs únicos del negocio
    - Datos de precio, stock, disponibilidad
    - Metadata comercial
    
    Returns:
        Lista de diccionarios con productos reales
    """
    print("📦 Extrayendo productos reales desde productos.csv...")
    products = parse_products_csv()
    
    # Validación básica
    valid_products = []
    for p in products:
        if not p.get('id'):
            print(f"   ⚠️ Producto sin ID, omitido: {p.get('title', 'N/A')}")
            continue
        
        if not p.get('title') or str(p.get('title')).strip() in ['', '0', 'nan', 'None', '.', '-']:
            print(f"   ⚠️ Producto sin título válido, omitido: ID={p.get('id')}")
            continue
        
        valid_products.append(p)
    
    print(f"   ✅ {len(valid_products)} productos válidos extraídos de productos.csv")
    return valid_products


# ═══════════════════════════════════════════════════════════════
# EXTRACCIÓN DE VADEMÉCUM (datos clínicos)
# ═══════════════════════════════════════════════════════════════

def extract_vademecum_from_csv() -> List[Dict]:
    """
    Extrae registros del Vademécum Clínico desde archivo CSV.
    
    Contiene datos médicos:
    - Especies (Perro, Gato, etc.)
    - Indicaciones médicas
    - Contraindicaciones
    - Dosificación
    
    Returns:
        Lista de diccionarios con datos del Vademécum
    """
    print("⚕️  Extrayendo datos clínicos desde Vademécum...")
    vademecum = parse_vademecum_csv()
    print(f"   ✅ {len(vademecum)} registros clínicos extraídos")
    return vademecum


# ═══════════════════════════════════════════════════════════════
# EXTRACCIÓN DE DATOS RELACIONALES
# ═══════════════════════════════════════════════════════════════

def extract_companies_from_csv() -> List[Dict]:
    """Extrae empresas/laboratorios desde CSV."""
    print("🏭 Extrayendo empresas...")
    companies = parse_companies_csv()
    print(f"   ✅ {len(companies)} empresas extraídas")
    return companies


def extract_categories_from_csv() -> List[Dict]:
    """Extrae categorías desde CSV."""
    print("📂 Extrayendo categorías...")
    categories = parse_categories_csv()
    print(f"   ✅ {len(categories)} categorías extraídas")
    return categories


def extract_offer_products_from_csv() -> List[Dict]:
    """
    Lee la tabla intermedia que vincula Offers <-> Products.
    """
    print("🔗 Extrayendo vínculos Ofertas-Productos...")
    links = parse_offerproducts_csv()
    print(f"   ✅ {len(links)} vínculos extraídos")
    return links


def extract_offers_from_csv() -> List[Dict]:
    """Extrae ofertas desde CSV."""
    print("🏷️  Extrayendo ofertas...")
    offers = parse_offers_csv()
    print(f"   ✅ {len(offers)} ofertas extraídas")
    return offers


# ═══════════════════════════════════════════════════════════════
# SINCRONIZACIÓN INCREMENTAL (API REST)
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# GESTIÓN DE TIMESTAMP DE SINCRONIZACIÓN
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# FUNCIONES DE UTILIDAD
# ═══════════════════════════════════════════════════════════════

def validate_extraction_results(data_dict: Dict[str, List]) -> bool:
    """
    Valida que la extracción haya sido exitosa.
    
    Args:
        data_dict: Diccionario con listas de datos extraídos
    
    Returns:
        True si hay datos válidos, False en caso contrario
    """
    has_data = False
    
    for key, data_list in data_dict.items():
        if data_list and len(data_list) > 0:
            has_data = True
            print(f"   ✅ {key}: {len(data_list)} registros")
        else:
            print(f"   ⚠️ {key}: Sin datos")
    
    return has_data