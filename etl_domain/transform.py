"""
Transformación: Generación de embeddings con IDENTIDAD COMPACTA
VERSIÓN 4.0: 
- Filtro de calidad mejorado
- Jerarquía de entidades (product/family/brand)
- Vinculación con ID real de productos.csv
- Metadata expandida para mejor retrieval
"""
import re
from typing import List, Dict, Set, Tuple
from unidecode import unidecode 

from etl_domain.product_parser import extract_product_metadata
from rag_domain.embeddings import get_embedding_model


# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE FILTRADO
# ═══════════════════════════════════════════════════════════════

# Keywords médicas cortas permitidas
MEDICAL_SHORT_TERMS = {
    'tos', 'ojo', 'ojos', 'piel', 'pus', 'uña', 'uñas', 'pie', 'pies',
    'ulcera', 'otitis', 'diarrea', 'vomito'
}

# Lista negra EXPANDIDA
GARBAGE_TITLES = {
    # Palabras genéricas
    'asas', 'test', 'prueba', 'vario', 'varios', 's/n', 's/d',
    'sin nombre', 'pendiente', 'generico', 'a confirmar',
    'bonificacion', 'descuento', 'regalo', 'promo', 'promocion',
    
    # Marcas solas (sin producto)
    'konig', 'labyes', 'bayer', 'zoetis', 'john martin', 'holliday',
    'brower', 'brouwer', 'over', 'ruminal', 'proagro', 'biogenesis',
    'bago', 'bagó', 'richmond', 'drag pharma', 'vetanco',
    
    # Nombres de empresas
    'laboratorio', 'laboratorios', 's.a.', 's.a', 'sa', 'srl',
    'argentina', 'veterinaria', 'pet', 'company', 'corp',
    
    # Términos ambiguos
    'otro', 'otros', 'mas', 'producto', 'item', 'articulo'
}

# Indicadores de productos reales (debe tener al menos uno)
REAL_PRODUCT_INDICATORS = {
    # Rangos de peso
    r'\d+\s*(?:a|hasta)\s*\d+\s*kg',
    
    # Presentaciones
    r'\d+\s*(?:mg|ml|gr|g|kg)',
    r'\bcomprimidos?\b', r'\btabletas?\b', r'\bgotas?\b',
    r'\bpipetas?\b', r'\binyectable\b', r'\bshampoo\b',
    r'\bspray\b', r'\bcollar\b', r'\bcrema\b', r'\bgel\b',
    
    # Unidades
    r'x\s*\d+', r'\d+\s*unidades'
}


# ═══════════════════════════════════════════════════════════════
# VALIDACIÓN DE CALIDAD (MEJORADA)
# ═══════════════════════════════════════════════════════════════

def validate_product_quality(product: Dict) -> Tuple[bool, str]:
    """
    Verifica si el producto tiene calidad suficiente para ser indexado.
    
    Returns:
        Tuple (is_valid, rejection_reason)
    """
    title = str(product.get('title', '')).strip()
    lab = str(product.get('enterprise_title', '')).strip()
    
    # Normalización
    norm_title = unidecode(title).lower()
    norm_lab = unidecode(lab).lower()
    
    # 1. Longitud mínima
    if len(norm_title) < 3:
        return False, "Título muy corto (<3 chars)"
    
    # 2. Lista negra directa
    if norm_title in GARBAGE_TITLES:
        return False, f"Lista negra: '{norm_title}'"
    
    # 3. Título == Laboratorio (marca sola)
    if norm_lab and norm_lab != 'n/a':
        if norm_title == norm_lab:
            return False, f"Título == Lab: '{norm_title}'"
    
    # 4. Solo marca (una palabra contenida en lab)
    title_words = norm_title.split()
    if len(title_words) == 1 and norm_lab:
        if title_words[0] in norm_lab:
            return False, f"Solo marca: '{title_words[0]}' ⊂ '{norm_lab}'"
    
    # 5. Palabras genéricas solas
    generic_single_words = {'producto', 'item', 'articulo', 'promo', 'oferta'}
    if len(title_words) == 1 and title_words[0] in generic_single_words:
        return False, f"Palabra genérica: '{title_words[0]}'"
    
    # 6. Verificar indicadores de producto real
    has_indicator = False
    for pattern in REAL_PRODUCT_INDICATORS:
        if re.search(pattern, title, re.IGNORECASE):
            has_indicator = True
            break
    
    # Si el título es corto (<15 chars) y no tiene indicadores, rechazar
    if len(title) < 15 and not has_indicator:
        # Excepciones: productos conocidos cortos
        known_short = {'nexgard', 'bravecto', 'simparica', 'advocate', 'frontline'}
        if not any(k in norm_title for k in known_short):
            return False, "Título corto sin indicadores de producto"
    
    return True, ""


def determine_entity_hierarchy(title: str, has_weight: bool, has_presentation: bool) -> str:
    """
    Determina el tipo de entidad basado en la especificidad.
    
    JERARQUÍA:
    - product: Producto específico (ej: "POWER GOLD DE 10 A 20 KG")
    - family: Familia de productos (ej: "POWER GOLD")
    - brand: Marca genérica (ej: "POWER")
    
    Args:
        title: Título del producto
        has_weight: Si tiene rango de peso
        has_presentation: Si tiene presentación específica
    
    Returns:
        'product', 'family', o 'brand'
    """
    word_count = len(title.split())
    
    # Producto específico: tiene peso o presentación
    if has_weight or has_presentation:
        return 'product'
    
    # Familia: 2-4 palabras sin detalles técnicos
    if 2 <= word_count <= 4:
        return 'family'
    
    # Marca: 1 palabra o muy genérico
    if word_count == 1:
        return 'brand'
    
    # Default: familia
    return 'family'


# ═══════════════════════════════════════════════════════════════
# EXTRACCIÓN DE KEYWORDS CLÍNICAS
# ═══════════════════════════════════════════════════════════════

def extract_clinical_keywords(text: str) -> Set[str]:
    """Extrae keywords clínicas únicas del texto."""
    if not text:
        return set()
    
    clean_text = unidecode(str(text)).lower()
    words = re.findall(r'\b[a-z0-9]+\b', clean_text)
    
    keywords = set()
    for word in words:
        if len(word) > 3 or word in MEDICAL_SHORT_TERMS:
            keywords.add(word)
    
    return keywords


# ═══════════════════════════════════════════════════════════════
# EXPANSIÓN DE TÉRMINOS VETERINARIOS
# ═══════════════════════════════════════════════════════════════

def expand_veterinary_terms(title: str) -> str:
    """Expande abreviaturas comunes y jerga técnica."""
    text = title.upper()
    
    replacements = {
        r'\bSH\.': 'SHAMPOO ', r'\bSH\b': 'SHAMPOO ',
        r'\bCOMP\.': 'COMPRIMIDOS ', r'\bCOMP\b': 'COMPRIMIDOS ',
        r'\bCS\b': 'COMPRIMIDOS ', r'\bTABL\b': 'TABLETAS ',
        r'\bINY\.': 'INYECTABLE ', r'\bINY\b': 'INYECTABLE ',
        r'\bS\.I\.': 'SOLUCION INYECTABLE ', r'\bJGA\.': 'JERINGA ',
        r'\bFCO\.': 'FRASCO ', r'\bAMPO?\.': 'AMPOLLA ',
        r'\bSOL\.': 'SOLUCION ', r'\bUNG\.': 'UNGUENTO ',
        r'\bPOMO\b': 'CREMA ', r'\bSUSP\.': 'SUSPENSION ',
        r'\bGT\.': 'GOTAS ', r'\bGTS\.': 'GOTAS ',
        r'\bPIP\.': 'PIPETA ', r'\bAER\b': 'AEROSOL ',
        r'\bPOUR ON\b': 'TOPICO DORSAL ',
        r'\bVAC\.': 'VACUNA ', r'\bVANG\b': 'VACUNA VANGUARD ',
        r'\bFELOCELL\b': 'VACUNA FELOCELL ',
        r'\bDEFENSOR\b': 'VACUNA ANTIRRABICA DEFENSOR ',
        r'\bCANIGEN\b': 'VACUNA CANIGEN ',
        r'\bVETSCAN\b': 'EQUIPO DIAGNOSTICO VETSCAN ',
        r'\bROTOR\b': 'ROTOR ANALISIS ', r'\bREAGENT\b': 'REACTIVO ',
        r'H/(\d)': r'HASTA \1', r'H-(\d)': r'HASTA \1',
        r'\bC/\b': 'CON ', r'\bS/\b': 'SIN ',
        r'\bPEQ\.': 'PEQUEÑOS ', r'\bMED\.': 'MEDIANOS ',
        r'\bGDE\.': 'GRANDES ', r'\bGDE\b': 'GRANDES ',
        r'\bOSS\.': 'OSSPRET ', r'\bHOLL\b': 'HOLLIDAY ',
        r'\bJ\.?\s?MARTIN\b': 'JOHN MARTIN ',
        r'\bHOSP\.': 'HOSPITALARIO ', r'\bHOSP\b': 'HOSPITALARIO ',
        r'\bL\.A\.': 'LARGA ACCION ', r'\bCURABICH\b': 'ANTIMIASICO CICATRIZANTE ',
        r'\bANTIP\b': 'ANTIPARASITARIO ',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    # Normalizar ceros
    text = re.sub(r'\b0+(\d)', r'\1', text)
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_special_tags(title_expanded: str) -> List[str]:
    """Extrae tags especiales del título expandido."""
    tags = []
    text = title_expanded.upper()
    
    if 'HOSPITALARIO' in text:
        tags.append('hospitalario')
    if 'VACUNA' in text or 'ANTIRRABICA' in text:
        tags.append('vacuna')
    if 'DIAGNOSTICO' in text or 'REACTIVO' in text:
        tags.append('insumo_diagnostico')
    if 'PEDIATRICO' in text or 'CACHORRO' in text or 'PUPPY' in text:
        tags.append('pediatrico')
    if 'GERIATRICO' in text or 'SENIOR' in text:
        tags.append('geriatrico')
    if 'TRAZABILIDAD' in text:
        tags.append('trazado')
    
    return tags


def normalize_for_filter(text: str) -> str:
    """Normaliza texto para filtros SQL."""
    if not text:
        return ""
    text_clean = unidecode(str(text)).lower()
    return re.sub(r'[^a-z0-9]', '', text_clean)


# ═══════════════════════════════════════════════════════════════
# GENERACIÓN DE TEXTO PARA EMBEDDINGS
# ═══════════════════════════════════════════════════════════════

def product_to_text(product: Dict) -> str:
    """
    Genera IDENTIDAD COMPACTA para el vector.
    
    Incluye:
    - Categoría
    - Título expandido
    - Laboratorio
    - Droga/Principio activo
    - Especies
    - Keywords clínicas
    """
    parts = []
    
    # Categoría
    cat = product.get('category_name', '')
    if cat:
        parts.append(cat)
    
    # Título expandido
    title = product.get('title', '')
    if title:
        expanded_title = expand_veterinary_terms(title)
        parts.append(expanded_title)
    
    # Laboratorio
    lab = product.get('enterprise_title', '')
    if lab:
        parts.append(f"Laboratorio {lab}")
    
    # Droga
    drug = product.get('active_ingredient', '')
    if drug:
        parts.append(f"Droga {drug}")
    
    # Especies
    species = product.get('species_data', [])
    if species:
        species_str = ' '.join(species)
        parts.append(f"Especies {species_str}")
    
    # Keywords clínicas
    clinical_keywords = set()
    
    med_ind = product.get('medical_indications', '')
    if med_ind:
        clinical_keywords.update(extract_clinical_keywords(med_ind))
    
    action = product.get('therapeutic_action', '')
    if action:
        clinical_keywords.update(extract_clinical_keywords(action))
    
    if clinical_keywords:
        parts.append(' '.join(sorted(clinical_keywords)))
    
    return ". ".join(parts)


# ═══════════════════════════════════════════════════════════════
# GENERACIÓN DE EMBEDDINGS (BATCH)
# ═══════════════════════════════════════════════════════════════

def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Genera embeddings en batch."""
    embedder = get_embedding_model()
    clean_texts = [re.sub(r'\s+', ' ', t).strip() for t in texts]
    return embedder.embed_documents(clean_texts)


# ═══════════════════════════════════════════════════════════════
# PREPARACIÓN DE EMBEDDINGS DE PRODUCTOS (MEJORADA)
# ═══════════════════════════════════════════════════════════════

def prepare_product_embeddings(products: List[Dict]) -> List[Dict]:
    """
    Prepara productos para vectorización con:
    - Filtro de calidad avanzado
    - Jerarquía de entidades
    - Vinculación con ID real
    - Metadata expandida
    """
    if not products:
        return []
    
    print(f"\n🕵️  Analizando calidad de {len(products)} productos...")
    print("   " + "="*70)
    
    # 1. FILTRADO DE BASURA
    valid_products = []
    skipped_count = 0
    rejection_reasons = {}
    
    for p in products:
        is_valid, reason = validate_product_quality(p)
        
        if is_valid:
            valid_products.append(p)
        else:
            skipped_count += 1
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            
            # Log detallado
            p_id = p.get('id', '?')
            p_title = p.get('title', 'Sin Título')
            p_lab = p.get('enterprise_title', 'N/A')
            print(f"   🚫 [RECHAZADO] ID:{p_id} | '{p_title}' | Lab: '{p_lab}' | Razón: {reason}")
    
    print("   " + "="*70)
    
    # Resumen de rechazos
    if rejection_reasons:
        print("\n   📊 RESUMEN DE RECHAZOS:")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"      - {reason}: {count}")
    
    if skipped_count > 0:
        print(f"\n   🧹 {skipped_count} productos eliminados ({100*skipped_count/len(products):.1f}%)")
    else:
        print("\n   ✅ Calidad perfecta: No se rechazó ningún producto.")
    
    if not valid_products:
        print("\n   ⚠️  No quedaron productos válidos para procesar.")
        return []
    
    print(f"\n   📄 Transformando {len(valid_products)} productos válidos...")
    
    # 2. GENERAR TEXTOS Y EMBEDDINGS
    texts = [product_to_text(p) for p in valid_products]
    embeddings = generate_embeddings_batch(texts)
    
    prepared = []
    
    for product, text, embedding in zip(valid_products, texts, embeddings):
        title = str(product.get('title', ''))
        
        # Expansión y Tags
        title_expanded = expand_veterinary_terms(title)
        special_tags = extract_special_tags(title_expanded)
        
        # Metadata Numérica
        parsed_metadata = extract_product_metadata(title)
        
        # Determinar jerarquía
        has_weight = bool(parsed_metadata.get('weight_range'))
        has_presentation = bool(parsed_metadata.get('presentation_normalized'))
        entity_hierarchy = determine_entity_hierarchy(title, has_weight, has_presentation)
        
        # Filtros SQL
        lab_clean = normalize_for_filter(product.get('enterprise_title'))
        cat_clean = normalize_for_filter(product.get('category_name'))
        drug_clean = normalize_for_filter(product.get('active_ingredient'))
        
        # Keywords expandidas
        keywords_set = set()
        keywords_set.add(lab_clean)
        keywords_set.add(cat_clean)
        keywords_set.add(drug_clean)
        
        for word in title_expanded.lower().split():
            clean_w = normalize_for_filter(word)
            if len(clean_w) > 2:
                keywords_set.add(clean_w)
        
        species_list = product.get('species_data', [])
        for sp in species_list:
            keywords_set.add(normalize_for_filter(sp))
        
        med_ind = product.get('medical_indications', '')
        for kw in extract_clinical_keywords(med_ind):
            keywords_set.add(kw)
        
        action = product.get('therapeutic_action', '')
        for kw in extract_clinical_keywords(action):
            keywords_set.add(kw)
        
        base_phrase = unidecode(title_expanded).lower().strip()
        keywords_str = f"{base_phrase} {' '.join(keywords_set)}"
        
        # METADATA EXPANDIDA
        metadata = {
            # IDs
            'product_id': int(product['id']),  # ← ID REAL de productos.csv
            'vademecum_id': product.get('vademecum_id'),  # ← ID del Vademécum (si existe)
            'vademecum_match_score': product.get('vademecum_match_score', 0.0),
            
            # Identidad
            'title': title[:200],
            'enterprise_title': str(product.get('enterprise_title', ''))[:100],
            'category': str(product.get('category_name', ''))[:50],
            
            # Jerarquía
            'entity_hierarchy': entity_hierarchy,  # ← product/family/brand
            
            # Droga y acción
            'drug': str(product.get('active_ingredient', ''))[:100],
            'action': str(product.get('therapeutic_action', ''))[:100],
            
            # Datos clínicos
            'medical_indications': str(product.get('medical_indications', ''))[:1000],
            'contraindications': str(product.get('contraindications', ''))[:500],
            'clinical_dosage': str(product.get('clinical_dosage', ''))[:300],
            'description': str(product.get('description', ''))[:500],
            
            # Filtros SQL
            'filter_lab': lab_clean,
            'filter_category': cat_clean,
            'filter_drug': drug_clean,
            'search_keywords': keywords_str,
            'species_filter': ' '.join([normalize_for_filter(s) for s in species_list]),
            
            # Tags
            'tags': special_tags,
            'is_vaccine': 'vacuna' in special_tags,
            'is_hospitalary': 'hospitalario' in special_tags,
            
            # Comercial
            'is_offer': False,
            'has_transfer': False,
            
            # Metadata numérica
            'dosage_value': parsed_metadata.get('dosage_value'),
            'dosage_unit': parsed_metadata.get('dosage_unit'),
            'weight_range': parsed_metadata.get('weight_range'),
            'presentation': parsed_metadata.get('presentation_normalized'),
        }
        
        prepared.append({
            'entity_type': 'product',
            'entity_id': int(product['id']),  # ← ID REAL
            'content_text': text,
            'embedding': embedding,
            'metadata': metadata
        })
    
    print(f"   ✅ {len(prepared)} embeddings generados")
    return prepared


# ═══════════════════════════════════════════════════════════════
# PREPARACIÓN DE OTROS TIPOS (OFERTAS, TRANSFERS, EMPRESAS)
# ═══════════════════════════════════════════════════════════════

def offer_to_text(offer: Dict) -> str:
    """Genera texto para ofertas."""
    parts = []
    title = expand_veterinary_terms(offer.get('title', ''))
    parts.append(f"Oferta {title}")
    if offer.get('product_name'):
        parts.append(offer['product_name'])
    lab = offer.get('enterprise_supplier_title')
    if lab:
        parts.append(f"Proveedor {lab}")
    if offer.get('active_ingredient'):
        parts.append(f"Droga {offer['active_ingredient']}")
    return ". ".join(parts)


def transfer_to_text(transfer: Dict) -> str:
    """Genera texto para transfers."""
    parts = []
    parts.append("Transfer Bonificación")
    title = expand_veterinary_terms(transfer.get('title', ''))
    parts.append(title)
    if transfer.get('product_name'):
        parts.append(transfer['product_name'])
    lab = transfer.get('enterprise_supplier_title')
    if lab:
        parts.append(f"Laboratorio {lab}")
    return ". ".join(parts)


def company_to_text(company: Dict) -> str:
    """Genera texto para empresas."""
    parts = []
    if company.get('title'):
        parts.append(f"Empresa {company['title']}")
    if company.get('description'):
        parts.append(company['description'])
    return ". ".join(parts)


def prepare_offer_embeddings(offers: List[Dict]) -> List[Dict]:
    """Prepara ofertas para vectorización."""
    if not offers:
        return []
    
    print(f"📄 Transformando {len(offers)} ofertas...")
    
    texts = [offer_to_text(o) for o in offers]
    embeddings = generate_embeddings_batch(texts)
    
    prepared = []
    for offer, text, embedding in zip(offers, texts, embeddings):
        title = str(offer.get('title', ''))
        title_expanded = expand_veterinary_terms(title)
        
        parsed = extract_product_metadata(title)
        special_tags = extract_special_tags(title_expanded)
        
        lab_clean = normalize_for_filter(offer.get('enterprise_supplier_title'))
        
        keywords_set = set()
        keywords_set.add(lab_clean)
        keywords_set.add(normalize_for_filter(offer.get('product_name')))
        for word in title_expanded.lower().split():
            clean_w = normalize_for_filter(word)
            if len(clean_w) > 2:
                keywords_set.add(clean_w)
        
        metadata = {
            'title': title[:200],
            'valid_until': str(offer.get('date_to', '')),
            'enterprise_title': str(offer.get('enterprise_supplier_title', '')),
            'description': str(offer.get('description', ''))[:300],
            'filter_lab': lab_clean,
            'search_keywords': " ".join(keywords_set),
            'tags': special_tags,
            'is_offer': True,
            'has_transfer': False,
            'weight_range': parsed.get('weight_range'),
            'presentation': parsed.get('presentation_normalized'),
            'entity_hierarchy': 'offer',
        }
        
        prepared.append({
            'entity_type': 'offer',
            'entity_id': int(offer['id']),
            'content_text': text,
            'embedding': embedding,
            'metadata': metadata
        })
    
    return prepared


def prepare_transfer_embeddings(transfers: List[Dict]) -> List[Dict]:
    """Prepara transfers para vectorización."""
    if not transfers:
        return []
    
    print(f"📄 Transformando {len(transfers)} transfers...")
    
    texts = [transfer_to_text(t) for t in transfers]
    embeddings = generate_embeddings_batch(texts)
    
    prepared = []
    for transfer, text, embedding in zip(transfers, texts, embeddings):
        title = str(transfer.get('title', ''))
        lab_clean = normalize_for_filter(transfer.get('enterprise_supplier_title'))
        keywords_set = set()
        keywords_set.add(lab_clean)
        keywords_set.add(normalize_for_filter(transfer.get('product_name')))
        
        metadata = {
            'title': title[:200],
            'supplier': str(transfer.get('enterprise_supplier_title', '')),
            'description': str(transfer.get('description', ''))[:300],
            'filter_lab': lab_clean,
            'search_keywords': " ".join(keywords_set),
            'has_transfer': True,
            'type': 'transfer_rule',
            'entity_hierarchy': 'transfer',
        }
        
        prepared.append({
            'entity_type': 'transfer',
            'entity_id': int(transfer['id']),
            'content_text': text,
            'embedding': embedding,
            'metadata': metadata
        })
    
    return prepared


def prepare_company_embeddings(companies: List[Dict]) -> List[Dict]:
    """Prepara empresas para vectorización."""
    if not companies:
        return []
    
    print(f"📄 Transformando {len(companies)} empresas...")
    
    texts = [company_to_text(c) for c in companies]
    embeddings = generate_embeddings_batch(texts)
    
    prepared = []
    for comp, text, emb in zip(companies, texts, embeddings):
        title = str(comp.get('title', ''))
        metadata = {
            'title': title[:200],
            'email': str(comp.get('email', ''))[:100],
            'description': str(comp.get('description', ''))[:300],
            'filter_lab': normalize_for_filter(title),
            'entity_hierarchy': 'company',
        }
        
        prepared.append({
            'entity_type': 'company',
            'entity_id': int(comp['id']),
            'content_text': text,
            'embedding': emb,
            'metadata': metadata
        })
    
    return prepared