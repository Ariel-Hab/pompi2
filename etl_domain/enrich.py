"""
etl_domain/enrich.py
Lógica de cruce de datos y enriquecimiento.
VERSIÓN 4.0: Cruce productos.csv (REAL) ⟷ Vademécum (CLÍNICO)
"""
import re
from typing import List, Dict, Tuple
from unidecode import unidecode


# ═══════════════════════════════════════════════════════════════
# UTILIDADES DE NORMALIZACIÓN
# ═══════════════════════════════════════════════════════════════

def normalize_id(value):
    """Normaliza IDs a string limpio."""
    if value is None or str(value).strip() == "":
        return ""
    try:
        clean_int = int(float(value))
        return str(clean_int)
    except (ValueError, TypeError):
        return str(value).strip()


def normalize_name_for_match(text: str) -> str:
    """
    Normaliza nombres para cruzar productos con vademécum.
    Elimina unidades incluso si están pegadas al número (ej: 500MG).
    
    Ejemplos:
    - "POWER GOLD DE 10 A 20 KG" → "powergoldde10a20"
    - "Bravecto 500mg" → "bravecto"
    """
    if not text:
        return ""
    
    # 1. Limpieza base
    clean = unidecode(str(text)).lower()
    
    # 2. Eliminar unidades (mg, ml, kg, etc.)
    unit_regex = r'(\b|(?<=\d))(mg|ml|kg|gr|g|cm|comp|comprimidos|tabletas)\b'
    clean = re.sub(unit_regex, '', clean)
    
    # 3. Eliminar palabras comunes que varían
    noise_words = ['de', 'a', 'x', 'para', 'con', 'en']
    for word in noise_words:
        clean = re.sub(r'\b' + word + r'\b', '', clean)
    
    # 4. Solo alfanuméricos
    return re.sub(r'[^a-z0-9]', '', clean)


def fuzzy_match_name(name1: str, name2: str) -> float:
    """
    Calcula similitud entre dos nombres normalizados.
    
    Returns:
        Score 0.0-1.0 (1.0 = match perfecto)
    """
    norm1 = normalize_name_for_match(name1)
    norm2 = normalize_name_for_match(name2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # Exact match
    if norm1 == norm2:
        return 1.0
    
    # Uno contiene al otro
    if norm1 in norm2 or norm2 in norm1:
        shorter = min(len(norm1), len(norm2))
        longer = max(len(norm1), len(norm2))
        return shorter / longer
    
    # Similitud de tokens
    tokens1 = set(norm1[i:i+3] for i in range(len(norm1)-2))
    tokens2 = set(norm2[i:i+3] for i in range(len(norm2)-2))
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# CRUCE PRODUCTOS.CSV ⟷ VADEMÉCUM
# ═══════════════════════════════════════════════════════════════

def enrich_products_with_vademecum(
    real_products: List[Dict],
    vademecum_rows: List[Dict],
    match_threshold: float = 0.7
) -> Tuple[int, List[Dict]]:
    """
    Cruza productos.csv (REAL) con Vademécum (CLÍNICO).
    
    ESTRATEGIA DE CRUCE:
    1. Match exacto por nombre normalizado
    2. Match fuzzy si no hay exacto (threshold > 0.7)
    3. Enriquecimiento solo si hay match confiable
    
    Args:
        real_products: Lista de productos de productos.csv
        vademecum_rows: Lista de registros del Vademécum
        match_threshold: Score mínimo para considerar un match (0.7 = 70%)
    
    Returns:
        Tuple (matches_count, unmatched_products)
    """
    print("🔗 Cruzando productos.csv ⟷ Vademécum...")
    print(f"   Productos reales: {len(real_products)}")
    print(f"   Registros Vademécum: {len(vademecum_rows)}")
    print(f"   Threshold de match: {match_threshold}")
    print()
    
    # 1. Crear índice del Vademécum
    vademecum_map = {}
    for row in vademecum_rows:
        raw_name = row.get('PRODUCTO', '')
        if not raw_name:
            continue
        
        key = normalize_name_for_match(raw_name)
        if key:
            # Guardar el mejor match (más completo)
            if key not in vademecum_map or len(raw_name) > len(vademecum_map[key].get('PRODUCTO', '')):
                vademecum_map[key] = row
    
    print(f"   📋 Índice Vademécum: {len(vademecum_map)} claves únicas")
    
    # 2. Cruzar productos
    exact_matches = 0
    fuzzy_matches = 0
    no_matches = 0
    unmatched_products = []
    
    for prod in real_products:
        prod_title = prod.get('title', '')
        prod_id = prod.get('id', '?')
        prod_key = normalize_name_for_match(prod_title)
        
        # Inicializar campos clínicos como vacíos
        prod['species_data'] = []
        prod['medical_indications'] = ''
        prod['therapeutic_action'] = ''
        prod['contraindications'] = ''
        prod['clinical_dosage'] = ''
        prod['vademecum_match_score'] = 0.0
        prod['vademecum_id'] = None
        
        if not prod_key:
            no_matches += 1
            unmatched_products.append(prod)
            continue
        
        # Intentar match exacto
        if prod_key in vademecum_map:
            v_data = vademecum_map[prod_key]
            _inject_clinical_data(prod, v_data)
            prod['vademecum_match_score'] = 1.0
            prod['vademecum_id'] = v_data.get('id')  # Si el Vademécum tiene IDs
            exact_matches += 1
            continue
        
        # Intentar match fuzzy
        best_match = None
        best_score = 0.0
        
        for v_key, v_data in vademecum_map.items():
            score = fuzzy_match_name(prod_key, v_key)
            if score > best_score and score >= match_threshold:
                best_score = score
                best_match = v_data
        
        if best_match:
            _inject_clinical_data(prod, best_match)
            prod['vademecum_match_score'] = best_score
            prod['vademecum_id'] = best_match.get('id')
            fuzzy_matches += 1
            print(f"   🔍 Fuzzy match ({best_score:.2f}): '{prod_title}' ⟷ '{best_match.get('PRODUCTO', '')}'")
        else:
            no_matches += 1
            unmatched_products.append(prod)
    
    # 3. Reporte
    total = len(real_products)
    print()
    print("   " + "="*60)
    print(f"   📊 RESULTADOS DEL CRUCE:")
    print(f"      ✅ Matches exactos:  {exact_matches:4d} ({100*exact_matches/total:.1f}%)")
    print(f"      🔍 Matches fuzzy:    {fuzzy_matches:4d} ({100*fuzzy_matches/total:.1f}%)")
    print(f"      ❌ Sin match:        {no_matches:4d} ({100*no_matches/total:.1f}%)")
    print("   " + "="*60)
    print()
    
    # Log de productos sin match (primeros 10)
    if unmatched_products and len(unmatched_products) <= 20:
        print("   ⚠️  Productos sin datos clínicos:")
        for p in unmatched_products[:10]:
            print(f"      - {p.get('title', 'N/A')} (ID: {p.get('id', '?')})")
        if len(unmatched_products) > 10:
            print(f"      ... y {len(unmatched_products)-10} más")
        print()
    
    total_enriched = exact_matches + fuzzy_matches
    return total_enriched, unmatched_products


def _inject_clinical_data(product: Dict, vademecum_data: Dict):
    """
    Inyecta datos clínicos del Vademécum en el producto.
    
    CAMPOS INYECTADOS:
    - species_data: Lista de especies
    - medical_indications: Síntomas + Diagnóstico
    - therapeutic_action: Acción terapéutica
    - contraindications: Contraindicaciones
    - clinical_dosage: Modo de uso + Dosificación
    """
    
    # 1. ESPECIES
    raw_species = vademecum_data.get('ESPECIE', '')
    if raw_species and raw_species not in ['0', 'N/A', 'None', '']:
        product['species_data'] = [s.strip() for s in raw_species.split(',') if s.strip()]
    
    # 2. INDICACIONES MÉDICAS (Síntomas + Diagnóstico)
    desc_parts = []
    
    sintomas = str(vademecum_data.get('Sintomas', '')).strip()
    if sintomas and sintomas not in ['0', 'N/A', 'None', '']:
        desc_parts.append(sintomas)
    
    diagnostico = str(vademecum_data.get('Diagnostico', '')).strip()
    if diagnostico and diagnostico not in ['0', 'N/A', 'None', '']:
        desc_parts.append(diagnostico)
    
    product['medical_indications'] = " ".join(desc_parts)
    
    # 3. ACCIÓN TERAPÉUTICA
    action_raw = str(vademecum_data.get('ACCION TERAPEUTICA', '')).strip()
    if action_raw and action_raw not in ['0', 'N/A', 'None', '']:
        product['therapeutic_action'] = action_raw
    
    # 4. CONTRAINDICACIONES
    contra_raw = str(vademecum_data.get('Contra indicaciones / Precauciones', '')).strip()
    if contra_raw and contra_raw not in ['0', 'N/A', 'None', '']:
        product['contraindications'] = contra_raw
    
    # 5. DOSIFICACIÓN CLÍNICA
    dosage_parts = []
    
    modo_uso = str(vademecum_data.get('Modo de usos', '')).strip()
    if modo_uso and modo_uso not in ['0', 'N/A', 'None', '']:
        dosage_parts.append(modo_uso)
    
    dosificacion = str(vademecum_data.get('Dosificación', '')).strip()
    if dosificacion and dosificacion not in ['0', 'N/A', 'None', '']:
        dosage_parts.append(dosificacion)
    
    product['clinical_dosage'] = ". ".join(dosage_parts)


# ═══════════════════════════════════════════════════════════════
# ENRIQUECIMIENTO CON EMPRESAS/LABORATORIOS
# ═══════════════════════════════════════════════════════════════

def enrich_data_with_companies(
    products: List[Dict],
    offers: List[Dict],
    companies: List[Dict]
) -> Tuple[int, int]:
    """Enriquece productos y ofertas con nombres de empresas."""
    print("🏭 Enriqueciendo con datos de empresas/laboratorios...")
    
    company_map = {
        normalize_id(c['id']): c['title'].strip()
        for c in companies
        if c.get('title')
    }
    
    # Productos
    prod_enriched = 0
    for p in products:
        emp_id = normalize_id(
            p.get('enterprise_id') or p.get('enterprise_supplier_product_id')
        )
        if emp_id in company_map:
            p['enterprise_title'] = company_map[emp_id]
            prod_enriched += 1
    
    # Ofertas
    offer_enriched = 0
    for o in offers:
        supp_id = normalize_id(
            o.get('enterprise_supplier_id') or o.get('enterprise_distributor_id')
        )
        if supp_id in company_map and 'enterprise_supplier_title' not in o:
            o['enterprise_supplier_title'] = company_map[supp_id]
            offer_enriched += 1
    
    print(f"   ✅ {prod_enriched} productos y {offer_enriched} ofertas enriquecidas")
    return prod_enriched, offer_enriched


# ═══════════════════════════════════════════════════════════════
# ENRIQUECIMIENTO CON CATEGORÍAS
# ═══════════════════════════════════════════════════════════════

def enrich_products_with_categories(
    products: List[Dict],
    categories: List[Dict]
) -> int:
    """Enriquece productos con nombres de categorías."""
    print("📂 Enriqueciendo con categorías...")
    
    cat_lookup = {
        normalize_id(c['id']): c['title'].strip()
        for c in categories
    }
    
    enriched = 0
    for prod in products:
        cat_id = normalize_id(prod.get('category_id'))
        if cat_id in cat_lookup:
            prod['category_name'] = cat_lookup[cat_id]
            enriched += 1
    
    print(f"   ✅ {enriched} productos categorizados")
    return enriched


# ═══════════════════════════════════════════════════════════════
# PROPAGACIÓN DE DATOS A OFERTAS/TRANSFERS
# ═══════════════════════════════════════════════════════════════

def enrich_items_with_product_details(
    commercial_items: List[Dict],
    products: List[Dict],
    offer_product_links: List[Dict]
) -> int:
    """
    Enriquece ofertas/transfers con detalles de productos.
    Heredan datos clínicos del producto padre.
    """
    print("🔗 Propagando datos de productos a ofertas/transfers...")
    
    # Índices
    prod_lookup = {
        normalize_id(p['id']): p
        for p in products
        if p.get('id')
    }
    
    offer_to_prod = {
        normalize_id(l['offer_id']): normalize_id(l['product_id'])
        for l in offer_product_links
    }
    
    enriched = 0
    for item in commercial_items:
        offer_id = normalize_id(item.get('id'))
        product_id = offer_to_prod.get(offer_id)
        
        if product_id and product_id in prod_lookup:
            product = prod_lookup[product_id]
            
            # Copiar campos básicos
            item['product_name'] = product.get('title', '')
            item['active_ingredient'] = product.get('active_ingredient', '')
            item['therapeutic_action'] = product.get('therapeutic_action', '')
            item['category_name'] = product.get('category_name', '')
            
            # Heredar datos clínicos del Vademécum
            item['species_data'] = product.get('species_data', [])
            item['medical_indications'] = product.get('medical_indications', '')
            item['contraindications'] = product.get('contraindications', '')
            item['clinical_dosage'] = product.get('clinical_dosage', '')
            item['vademecum_match_score'] = product.get('vademecum_match_score', 0.0)
            
            # Laboratorio (si no tiene)
            if not item.get('enterprise_supplier_title'):
                item['enterprise_supplier_title'] = product.get('enterprise_title', '')
            
            enriched += 1
    
    print(f"   ✅ {enriched} items comerciales enriquecidos")
    return enriched