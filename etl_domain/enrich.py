"""
etl_domain/enrich.py
Lógica de cruce de datos y enriquecimiento.
VERSIÓN 3.0: Asegura integridad de datos clínicos del Vademécum
"""
import re
from typing import List, Dict
from unidecode import unidecode

def normalize_id(value):
    """Normaliza IDs a string limpio."""
    if value is None or str(value).strip() == "": return ""
    try:
        clean_int = int(float(value))
        return str(clean_int)
    except (ValueError, TypeError):
        return str(value).strip()

def normalize_name_for_match(text: str) -> str:
    """
    Normaliza nombres para cruzar productos con vademécum.
    Elimina unidades incluso si están pegadas al número (ej: 500MG).
    """
    if not text: return ""
    
    # 1. Limpieza base
    clean = unidecode(str(text)).lower()
    
    # 2. Eliminar unidades
    unit_regex = r'(\b|(?<=\d))(mg|ml|kg|gr|g|cm|comp|comprimidos)\b'
    clean = re.sub(unit_regex, '', clean)
    
    # 3. Eliminar todo lo que no sea alfanumérico
    return re.sub(r'[^a-z0-9]', '', clean)

def enrich_products_with_vademecum(products: List[Dict], vademecum_rows: List[Dict]) -> int:
    """
    Cruza la lista de productos con las filas del CSV Vademécum.
    Usa el nombre del producto ('PRODUCTO' vs 'title') como llave de cruce.
    
    CAMPOS AGREGADOS (Asegura integridad):
    - species_data: Lista de especies (Perro, Gato, etc)
    - medical_indications: Síntomas + Diagnóstico (texto completo)
    - therapeutic_action: Acción terapéutica
    - contraindications: Contraindicaciones y precauciones
    - clinical_dosage: Modo de uso + dosificación
    """
    print("⚕️  Enriqueciendo con datos clínicos del Vademécum...")
    
    # 1. Crear índice del Vademécum por nombre normalizado
    vademecum_map = {}
    for row in vademecum_rows:
        raw_name = row.get('PRODUCTO', '')
        key = normalize_name_for_match(raw_name)
        if key:
            vademecum_map[key] = row

    matches = 0
    
    # 2. Iterar productos y buscar coincidencias
    for prod in products:
        prod_title = prod.get('title', '')
        key = normalize_name_for_match(prod_title)
        
        if key in vademecum_map:
            v_data = vademecum_map[key]
            
            # --- INYECCIÓN DE DATOS CLÍNICOS (ASEGURAR INTEGRIDAD) ---
            
            # 1. ESPECIES (Normalizamos la lista separada por comas)
            raw_species = v_data.get('ESPECIE', '')
            if raw_species:
                prod['species_data'] = [s.strip() for s in raw_species.split(',') if s.strip()]
            else:
                prod['species_data'] = []
            
            # 2. INDICACIONES MÉDICAS (Combinar Síntomas + Diagnóstico)
            desc_parts = []
            
            sintomas = v_data.get('Sintomas', '').strip()
            if sintomas and sintomas not in ['0', 'N/A', 'None']:
                desc_parts.append(sintomas)
            
            diagnostico = v_data.get('Diagnostico', '').strip()
            if diagnostico and diagnostico not in ['0', 'N/A', 'None']:
                desc_parts.append(diagnostico)
            
            prod['medical_indications'] = " ".join(desc_parts) if desc_parts else ""
            
            # 3. ACCIÓN TERAPÉUTICA (Asegurar que exista)
            action_raw = v_data.get('ACCION TERAPEUTICA', '').strip()
            if action_raw and action_raw not in ['0', 'N/A', 'None']:
                prod['therapeutic_action'] = action_raw
            else:
                prod['therapeutic_action'] = ""
            
            # 4. CONTRAINDICACIONES
            contra_raw = v_data.get('Contra indicaciones / Precauciones', '').strip()
            if contra_raw and contra_raw not in ['0', 'N/A', 'None']:
                prod['contraindications'] = contra_raw
            else:
                prod['contraindications'] = ""
            
            # 5. DOSIFICACIÓN CLÍNICA (Modo de uso + Dosificación)
            dosage_parts = []
            
            modo_uso = v_data.get('Modo de usos', '').strip()
            if modo_uso and modo_uso not in ['0', 'N/A', 'None']:
                dosage_parts.append(modo_uso)
            
            dosificacion = v_data.get('Dosificación', '').strip()
            if dosificacion and dosificacion not in ['0', 'N/A', 'None']:
                dosage_parts.append(dosificacion)
            
            prod['clinical_dosage'] = ". ".join(dosage_parts) if dosage_parts else ""
            
            matches += 1
    
    print(f"   ✅ {matches} productos enriquecidos con datos clínicos")
    return matches


# --- FUNCIONES EXISTENTES (Mantenidas para compatibilidad) ---

def enrich_data_with_companies(products, offers, companies):
    """Enriquece productos y ofertas con nombres de empresas."""
    print("🔗 Iniciando enriquecimiento de empresas...")
    company_map = {normalize_id(c['id']): c['title'].strip() for c in companies if c.get('title')}
    
    prod_enriched = 0
    for p in products:
        emp_id = normalize_id(p.get('enterprise_id') or p.get('enterprise_supplier_product_id'))
        if emp_id in company_map:
            p['enterprise_title'] = company_map[emp_id]
            prod_enriched += 1
    
    offer_enriched = 0        
    for o in offers:
        supp_id = normalize_id(o.get('enterprise_supplier_id') or o.get('enterprise_distributor_id'))
        if supp_id in company_map and 'enterprise_supplier_title' not in o: 
            o['enterprise_supplier_title'] = company_map[supp_id]
            offer_enriched += 1
    
    print(f"   ✅ {prod_enriched} productos y {offer_enriched} ofertas enriquecidas")
    return prod_enriched, offer_enriched

def enrich_products_with_categories(products, categories):
    """Enriquece productos con nombres de categorías."""
    print("📂 Enriqueciendo con categorías...")
    cat_lookup = {normalize_id(c['id']): c['title'].strip() for c in categories}
    
    enriched = 0
    for prod in products:
        cat_id = normalize_id(prod.get('category_id'))
        if cat_id in cat_lookup:
            prod['category_name'] = cat_lookup[cat_id]
            enriched += 1
    
    print(f"   ✅ {enriched} productos categorizados")
    return enriched

def enrich_items_with_product_details(commercial_items, products, offer_product_links):
    """Enriquece ofertas/transfers con detalles de productos."""
    print("🔍 Enriqueciendo items comerciales con detalles de productos...")
    
    prod_lookup = {normalize_id(p['id']): p for p in products if p.get('id')}
    offer_to_prod = {normalize_id(l['offer_id']): normalize_id(l['product_id']) for l in offer_product_links}

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
            
            # Laboratorio (si no tiene)
            if not item.get('enterprise_supplier_title'):
                item['enterprise_supplier_title'] = product.get('enterprise_title', '')
            
            enriched += 1
    
    print(f"   ✅ {enriched} items comerciales enriquecidos")
    return enriched