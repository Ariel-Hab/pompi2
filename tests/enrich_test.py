from etl_domain.enrich import enrich_products_with_vademecum, normalize_name_for_match
import pytest

# --- A. Tests de Normalización de Claves ---

def test_normalize_name_for_match_robustness():
    """
    Prueba crítica: El nombre en tu DB y en el Vademécum nunca son idénticos.
    La función debe ser agresiva normalizando.
    """
    # Caso 1: Espacios y Puntos
    db_name = "Amox. Plus 500 mg."
    vademecum_name = "AMOX PLUS 500MG" # Sin espacios ni puntos
    assert normalize_name_for_match(db_name) == normalize_name_for_match(vademecum_name)
    
    # Caso 2: Acentos y Mayúsculas
    db_name = "Ácido Hialurónico"
    vademecum_name = "acido hialuronico"
    assert normalize_name_for_match(db_name) == normalize_name_for_match(vademecum_name)
    
    # Caso 3: Caracteres especiales
    db_name = "Shampoo (Perros)"
    vademecum_name = "SHAMPOO PERROS"
    assert normalize_name_for_match(db_name) == normalize_name_for_match(vademecum_name)

# --- B. Test de Lógica de Enriquecimiento (Mocking Completo) ---

def test_enrichment_logic_completeness():
    # 1. Setup de Datos Mock
    products = [
        {'title': 'Apoquel 16mg', 'id': 1},     # Match exacto
        {'title': 'Rimadyl 100', 'id': 2},      # Match parcial normalizado
        {'title': 'Producto Desconocido', 'id': 3} # No match
    ]
    
    vademecum_rows = [
        {
            'PRODUCTO': 'APOQUEL 16MG',
            'ESPECIE': 'Caninos',
            'Sintomas': 'Prurito',
            'Diagnostico': 'Dermatitis atópica',
            'Contra indicaciones / Precauciones': 'No usar en menores de 12 meses',
            'Dosificación': '0.4-0.6 mg/kg',
            'Modo de usos': 'Oral'
        },
        {
            'PRODUCTO': 'RIMADYL 100 MG', # Note el MG extra
            'ESPECIE': 'Caninos',
            'Sintomas': 'Dolor e inflamación',
            # Falta Diagnostico, Falta Contraindicaciones
            'Dosificación': '4.4 mg/kg',
            'Modo de usos': ''
        }
    ]
    
    # 2. Ejecución
    matches = enrich_products_with_vademecum(products, vademecum_rows)
    
    # 3. Validaciones
    assert matches == 2
    
    # Producto 1: Match Completo
    p1 = products[0]
    assert p1['species_data'] == ['Caninos']
    assert "Prurito" in p1['medical_indications']
    assert "Dermatitis atópica" in p1['medical_indications'] # Unió Sintomas + Diagnostico
    assert "No usar en menores" in p1['contraindications']
    assert "Oral. 0.4-0.6 mg/kg" == p1['clinical_dosage'] # Unió Modo + Dosis
    
    # Producto 2: Match Normalizado y Datos Parciales
    p2 = products[1]
    assert "Dolor e inflamación" in p2['medical_indications']
    assert p2['contraindications'] == '' # Manejo de nulos
    
    # Producto 3: Sin cambios
    p3 = products[2]
    assert 'medical_indications' not in p3 # No se inyectaron llaves