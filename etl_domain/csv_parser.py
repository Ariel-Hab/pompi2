"""
Parser de archivos CSV para carga inicial
"""
import pandas as pd
from typing import List, Dict
from pathlib import Path
from core.config import ETL_CONFIG
import csv


def parse_products_csv(csv_path: str = None) -> List[Dict]:
    """
    Parsea archivo CSV de productos
    
    Args:
        csv_path: Ruta al archivo CSV (usa default si no se especifica)
    
    Returns:
        Lista de productos parseados
    
    Formato esperado del CSV (columnas):
    - id: int
    - title: str
    - description: str
    - active_ingredient: str
    - therapeutic_action: str
    - enterprise_title: str
    - enterprise_id: int
    - date: str (formato YYYY-MM-DD)
    - timestamp: str (formato ISO)
    """
    if csv_path is None:
        csv_path = ETL_CONFIG['csv_products_path']
    
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        print(f"⚠️  Archivo CSV no encontrado: {csv_path}")
        return []
    
    print(f"📄 Parseando CSV: {csv_path}")
    
    try:
        # Leer CSV con pandas
        df = pd.read_csv(csv_file, encoding='utf-8')
        
        # Validar columnas requeridas
        required_columns = ['id', 'title', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️  Columnas faltantes: {missing_columns}")
            print(f"   Columnas encontradas: {list(df.columns)}")
            return []
        
        # Reemplazar NaN con strings vacíos
        df = df.fillna('')
        
        # Convertir a lista de diccionarios
        products = df.to_dict('records')
        
        print(f"✅ Parseados {len(products)} productos del CSV")
        return products
        
    except Exception as e:
        print(f"❌ Error parseando CSV: {e}")
        return []


def parse_offers_csv(csv_path: str = None) -> List[Dict]:
    """
    Parsea archivo CSV de ofertas
    
    Args:
        csv_path: Ruta al archivo CSV
    
    Returns:
        Lista de ofertas parseadas
    
    Formato esperado del CSV (columnas):
    - id: int
    - title: str
    - description: str
    - status: int
    - date_from: str
    - date_to: str
    - enterprise_supplier_title: str
    - enterprise_distributor_title: str
    - timestamp: str
    """
    if csv_path is None:
        csv_path = ETL_CONFIG['csv_offers_path']
    
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        print(f"⚠️  Archivo CSV de ofertas no encontrado: {csv_path}")
        return []
    
    print(f"📄 Parseando CSV de ofertas: {csv_path}")
    
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        
        required_columns = ['id', 'title', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️  Columnas faltantes: {missing_columns}")
            return []
        
        df = df.fillna('')
        offers = df.to_dict('records')
        
        print(f"✅ Parseadas {len(offers)} ofertas del CSV")
        return offers
        
    except Exception as e:
        print(f"❌ Error parseando CSV: {e}")
        return []
    
def parse_offerproducts_csv(csv_path: str = None) -> List[Dict]:
    """
    Parsea archivo CSV que vincula Offers con Products
    
    Args:
        csv_path: Ruta al archivo CSV
    
    Returns:
        Lista de relaciones oferta-producto parseadas
    
    Formato esperado del CSV (columnas):
    - offer_id: int
    - product_id: int
    """
    if csv_path is None:
        csv_path = ETL_CONFIG.get('csv_offerproducts_path', '/app/data/offer_products.csv')
    
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        print(f"⚠️  Archivo CSV de oferta-producto no encontrado: {csv_path}")
        return []
    
    print(f"📄 Parseando CSV de oferta-producto: {csv_path}")
    
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        
        required_columns = ['offer_id', 'product_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️  Columnas faltantes: {missing_columns}")
            return []
        
        df = df.fillna('')
        relations = df.to_dict('records')
        
        print(f"✅ Parseadas {len(relations)} relaciones oferta-producto del CSV")
        return relations
        
    except Exception as e:
        print(f"❌ Error parseando CSV de oferta-producto: {e}")
        return []
def parse_vademecum_csv(csv_path: str = None) -> List[Dict]:
    if csv_path is None:
        csv_path = ETL_CONFIG.get('csv_vademecum_path', '/app/data/vademecum.csv')
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"⚠️  Archivo CSV de Vademécum no encontrado: {csv_path}")
        return []
    
    print(f"📄 Parseando Vademécum: {csv_path}")
    
    try:
        # 1. Detectar separador automáticamente
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            sample = f.read(2048) # Leer primeros 2kb
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=[',', ';', '\t'])
            separator = dialect.delimiter
            
        print(f"   ℹ️  Separador detectado: '{separator}'")

        # 2. Leer con el separador correcto
        df = pd.read_csv(csv_file, sep=separator, encoding='utf-8-sig')
        
        # Validación de columnas
        if 'PRODUCTO' not in df.columns:
            # Fallback por si el header tiene mayúsculas/minúsculas distintas
            cols_upper = {c: c.upper() for c in df.columns}
            df = df.rename(columns=cols_upper)
            
            if 'PRODUCTO' not in df.columns:
                print("⚠️  Error: Columna 'PRODUCTO' no encontrada.")
                return []

        df = df.fillna('')
        records = df.to_dict('records')
        print(f"✅ Parseados {len(records)} registros del Vademécum")
        return records
        
    except Exception as e:
        print(f"❌ Error crítico parseando Vademécum: {e}")
        return []
     
def parse_categories_csv(csv_path: str = None) -> List[Dict]:
    """
    Parsea archivo CSV de categorías
    
    Args:
        csv_path: Ruta al archivo CSV
    
    Returns:
        Lista de categorías parseadas
    
    Formato esperado del CSV (columnas):
    - id: int
    - title: str
    """
    if csv_path is None:
        csv_path = ETL_CONFIG.get('csv_categories_path', '/app/data/categories.csv')
    
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        print(f"⚠️  Archivo CSV de categorías no encontrado: {csv_path}")
        return []
    
    print(f"📄 Parseando CSV de categorías: {csv_path}")
    
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        
        required_columns = ['id', 'title']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️  Columnas faltantes: {missing_columns}")
            return []
        
        df = df.fillna('')
        categories = df.to_dict('records')
        
        print(f"✅ Parseadas {len(categories)} categorías del CSV")
        return categories
        
    except Exception as e:
        print(f"❌ Error parseando CSV de categorías: {e}")
        return []


def generate_sample_csv():
    """
    Genera archivos CSV de ejemplo para testing
    """
    # CSV de productos
    products_data = {
        'id': [1, 2, 3],
        'title': [
            'Ibuprofeno 600mg',
            'Amoxicilina 500mg',
            'Paracetamol 1g'
        ],
        'description': [
            'Antiinflamatorio no esteroideo para dolor y fiebre',
            'Antibiótico de amplio espectro',
            'Analgésico y antipirético'
        ],
        'active_ingredient': [
            'Ibuprofeno',
            'Amoxicilina',
            'Paracetamol'
        ],
        'therapeutic_action': [
            'Analgésico, antipirético y antiinflamatorio',
            'Antibacteriano',
            'Analgésico y antipirético'
        ],
        'enterprise_title': [
            'Laboratorio ABC',
            'Farmacia XYZ',
            'Laboratorio DEF'
        ],
        'enterprise_id': [5, 3, 7],
        'date': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'timestamp': [
            '2024-01-15T10:30:00',
            '2024-01-16T14:20:00',
            '2024-01-17T09:15:00'
        ]
    }
    
    # CSV de ofertas
    offers_data = {
        'id': [1, 2],
        'title': [
            'Descuento 20% en antibióticos',
            'Promo 2x1 analgésicos'
        ],
        'description': [
            'Oferta válida para todos los antibióticos de marca',
            'Compra dos unidades y paga una'
        ],
        'status': [1, 1],
        'date_from': ['2024-01-01', '2024-02-01'],
        'date_to': ['2024-12-31', '2024-02-28'],
        'enterprise_supplier_title': ['Laboratorio ABC', 'Farmacia XYZ'],
        'enterprise_distributor_title': ['Distribuidor A', 'Distribuidor B'],
        'timestamp': ['2024-01-01T00:00:00', '2024-02-01T00:00:00']
    }
    
    # Crear DataFrames
    df_products = pd.DataFrame(products_data)
    df_offers = pd.DataFrame(offers_data)
    
    # Guardar CSVs
    products_path = Path(ETL_CONFIG['csv_products_path'])
    offers_path = Path(ETL_CONFIG['csv_offers_path'])
    
    products_path.parent.mkdir(exist_ok=True)
    offers_path.parent.mkdir(exist_ok=True)
    
    df_products.to_csv(products_path, index=False, encoding='utf-8')
    df_offers.to_csv(offers_path, index=False, encoding='utf-8')
    
    print(f"✅ Generado CSV de productos en: {products_path}")
    print(f"✅ Generado CSV de ofertas en: {offers_path}")

def parse_companies_csv(csv_path: str = None) -> List[Dict]:
    """
    Parsea archivo CSV de empresas (Solo Proveedores)
    """
    if csv_path is None:
        # Asegúrate de tener esta key en tu ETL_CONFIG o usa un path default
        csv_path = ETL_CONFIG.get('csv_companies_path', '/app/data/enterprises.csv')
    
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        print(f"⚠️  Archivo CSV de empresas no encontrado: {csv_path}")
        return []
    
    print(f"📄 Parseando CSV de empresas: {csv_path}")
    
    try:
        # Leer CSV con pandas
        df = pd.read_csv(csv_file, encoding='utf-8')
        
        # 1. Validar columna necesaria para el filtro
        if 'enterprise_type_id' not in df.columns:
            print("⚠️  Columna 'enterprise_type_id' no encontrada. No se puede filtrar por proveedor.")
            return []

        # 2. FILTRO: Solo Proveedores (Type == 1)
        # Convertimos a numérico por si viene como string "1"
        df['enterprise_type_id'] = pd.to_numeric(df['enterprise_type_id'], errors='coerce').fillna(0).astype(int)
        
        initial_count = len(df)
        
        # AQUI ESTA LA CLAVE: Nos quedamos solo con las de tipo 1
        df = df[df['enterprise_type_id'] == 1]
        
        filtered_count = len(df)
        print(f"🔍 Filtro Proveedores: Se mantuvieron {filtered_count} de {initial_count} empresas.")

        # 3. Limpieza y Retorno
        df = df.fillna('')
        
        # Convertir a lista de diccionarios
        companies = df.to_dict('records')
        
        return companies
        
    except Exception as e:
        print(f"❌ Error parseando CSV de empresas: {e}")
        return []


def csv_stats(csv_path: str):
    """
    Muestra estadísticas de un archivo CSV
    
    Args:
        csv_path: Ruta al CSV
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        print(f"\n📊 Estadísticas de {csv_path}")
        print(f"  • Filas: {len(df)}")
        print(f"  • Columnas: {len(df.columns)}")
        print(f"  • Columnas disponibles: {list(df.columns)}")
        print(f"\n  Primeras 3 filas:")
        print(df.head(3).to_string())
        
    except Exception as e:
        print(f"❌ Error leyendo CSV: {e}")