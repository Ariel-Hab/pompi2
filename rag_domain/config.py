"""
Configuración V2 - Actualizada para Soft Filtering y Scoring Híbrido

CAMBIOS vs V1:
- Nuevos parámetros para proximity scoring
- Configuración de dynamic drop-off
- Eliminación de parámetros obsoletos (reranker)
"""

# ============================================================================
# CONFIGURACIÓN DE BÚSQUEDA VECTORIAL
# ============================================================================

# Número de resultados MÁXIMO a retornar (puede ser menos con drop-off)
DEFAULT_TOP_K = 6

# DEPRECADO: Ya no usamos buffer + reranker LLM
# RERANKING_BUFFER = 20  # ← Eliminado

# Threshold mínimo de similitud semántica (bajado para soft filtering)
SEMANTIC_THRESHOLD = 0.25  # Bajado de 0.35 a 0.25

# Pesos para el boost de coincidencias exactas en metadata
BOOST_WEIGHTS = {
    'title': 0.6,           # Coincide el nombre del producto
    'enterprise': 0.5,      # Coincide el laboratorio
    'drug': 0.4,           # Coincide la droga/principio activo
    'category': 0.3,       # Coincide la categoría
    'action': 0.25         # Coincide la acción terapéutica
}

# ============================================================================
# CONFIGURACIÓN DE PROXIMITY SCORING (NUEVO)
# ============================================================================

# Peso máximo para proximity score numérico
PROXIMITY_MAX_SCORE = 0.3

# Fórmula de proximity: PROXIMITY_MAX_SCORE / (1 + ABS(target - actual))
# Ejemplos con target=10mg:
#   - actual=10mg  → 0.3 / 1 = 0.30
#   - actual=11mg  → 0.3 / 2 = 0.15
#   - actual=20mg  → 0.3 / 11 = 0.027
#   - actual=50mg  → 0.3 / 41 = 0.007

# Pesos relativos en scoring total
SCORING_WEIGHTS = {
    'semantic': 0.6,    # Vector similarity (más importante)
    'proximity': 0.4    # Numerical proximity (complementario)
}

# ============================================================================
# CONFIGURACIÓN DE DYNAMIC DROP-OFF (NUEVO)
# ============================================================================

# Ratio de drop-off: resultados < (top_score * ratio) se descartan
DYNAMIC_DROPOFF_RATIO = 0.65  # 65% del top score

# Ejemplo:
# Si top_score = 0.90:
#   - Threshold = 0.90 * 0.65 = 0.585
#   - Productos con score < 0.585 → descartados
# Si top_score = 0.40 (búsqueda difícil):
#   - Threshold = 0.40 * 0.65 = 0.26
#   - Productos con score < 0.26 → descartados
#   (más permisivo para evitar listas vacías)

# Mínimo de resultados a retornar (override drop-off si es necesario)
MIN_RESULTS_OVERRIDE = 2

# Si después del drop-off quedan < MIN_RESULTS_OVERRIDE,
# retornar al menos MIN_RESULTS_OVERRIDE (si existen candidatos)

# ============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN DE QUERIES
# ============================================================================

# Keywords que indican búsqueda de ofertas
OFFER_KEYWORDS = [
    "oferta", "off", "desc", "promo", "promocion", 
    "descuento", "rebaja", "liquidacion"
]

# Keywords que indican búsqueda de transfers/bonificaciones
TRANSFER_KEYWORDS = [
    "transfer", "bonif", "regalo", "regla", 
    "+", "combo", "pack", "llevando", "bm"
]

# Frases a limpiar del query
STOP_PHRASES = [
    "que productos vende", "quien vende", "quien tiene", "que tiene",
    "busco", "necesito", "precio de", "tenes", "info de", "productos de",
    "informacion de", "dame", "quiero", "me das", "consulta sobre",
    "decime", "contame", "mostrame"
]

# ============================================================================
# CONFIGURACIÓN DE HISTORIAL Y CONTEXTO
# ============================================================================

# Número de mensajes recientes a incluir en el contexto del LLM
HISTORY_MESSAGE_LIMIT = 3

# Máxima longitud del texto de un producto en el contexto RAG
MAX_CONTENT_LENGTH_IN_CONTEXT = 400

# Peso del context boost (historial de búsqueda)
CONTEXT_BOOST_WEIGHT = 0.15

# ============================================================================
# CONFIGURACIÓN DE RESPUESTAS
# ============================================================================

# Tipos de intención válidos
VALID_INTENTS = ["SEARCH", "RECOMMENDATION", "SMALLTALK", "OUT_OF_SCOPE"]

# Mapeo de intenciones a top_k personalizado
INTENT_TOP_K_MAP = {
    "SEARCH": 8,
    "RECOMMENDATION": 4,
    "SMALLTALK": 3,
    "OUT_OF_SCOPE": 3
}

# ============================================================================
# CONFIGURACIÓN DE FUZZY MATCHING (NER)
# ============================================================================

# Cutoff de difflib para fuzzy matching (0.0 a 1.0)
FUZZY_MATCH_CUTOFF = 0.85

# Longitud mínima de palabra para aplicar fuzzy matching
FUZZY_MIN_WORD_LENGTH = 4

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def get_top_k_for_intent(intent: str) -> int:
    """Retorna el top_k apropiado según la intención."""
    return INTENT_TOP_K_MAP.get(intent, DEFAULT_TOP_K)


def get_search_params(intent: str = "SEARCH") -> dict:
    """
    Retorna diccionario con todos los parámetros de búsqueda.
    
    DEPRECADO: buffer ya no existe
    """
    return {
        "top_k": get_top_k_for_intent(intent),
        "threshold": SEMANTIC_THRESHOLD,
        "boost_weights": BOOST_WEIGHTS,
        "proximity_max": PROXIMITY_MAX_SCORE,
        "dropoff_ratio": DYNAMIC_DROPOFF_RATIO
    }


def get_proximity_score(target: float, actual: float) -> float:
    """
    Calcula proximity score para valor numérico.
    
    Esta función está duplicada aquí para fácil acceso desde config.
    La implementación principal está en search_v2.py
    """
    if actual is None or target is None:
        return 0.0
    
    difference = abs(target - actual)
    
    if difference == 0:
        return PROXIMITY_MAX_SCORE
    
    raw_score = PROXIMITY_MAX_SCORE / (1 + difference)
    return min(raw_score, PROXIMITY_MAX_SCORE)


# ============================================================================
# VALIDACIÓN DE CONFIGURACIÓN
# ============================================================================

def validate_config():
    """Valida que la configuración sea coherente."""
    
    # Threshold debe estar entre 0 y 1
    assert 0.0 <= SEMANTIC_THRESHOLD <= 1.0, \
        f"SEMANTIC_THRESHOLD ({SEMANTIC_THRESHOLD}) debe estar entre 0.0 y 1.0"
    
    # Boost weights deben estar entre 0 y 1
    assert all(0.0 <= v <= 1.0 for v in BOOST_WEIGHTS.values()), \
        "Todos los BOOST_WEIGHTS deben estar entre 0.0 y 1.0"
    
    # Proximity max debe ser positivo
    assert PROXIMITY_MAX_SCORE > 0, \
        f"PROXIMITY_MAX_SCORE ({PROXIMITY_MAX_SCORE}) debe ser > 0"
    
    # Scoring weights deben sumar <= 1
    total_weight = sum(SCORING_WEIGHTS.values())
    assert total_weight <= 1.0, \
        f"SCORING_WEIGHTS suma {total_weight}, debe ser <= 1.0"
    
    # Drop-off ratio debe estar entre 0 y 1
    assert 0.0 < DYNAMIC_DROPOFF_RATIO <= 1.0, \
        f"DYNAMIC_DROPOFF_RATIO ({DYNAMIC_DROPOFF_RATIO}) debe estar entre 0.0 y 1.0"
    
    # History limit debe ser positivo
    assert HISTORY_MESSAGE_LIMIT >= 2, \
        f"HISTORY_MESSAGE_LIMIT ({HISTORY_MESSAGE_LIMIT}) debe ser >= 2"
    
    # Fuzzy cutoff debe estar entre 0 y 1
    assert 0.0 <= FUZZY_MATCH_CUTOFF <= 1.0, \
        f"FUZZY_MATCH_CUTOFF ({FUZZY_MATCH_CUTOFF}) debe estar entre 0.0 y 1.0"
    
    print("✅ Configuración V2 validada correctamente")


# Ejecutar validación al importar
validate_config()


# ============================================================================
# GUÍA DE TUNING
# ============================================================================

"""
GUÍA DE TUNING DE PARÁMETROS:

1. SEMANTIC_THRESHOLD (actualmente 0.25):
   - Aumentar (ej: 0.35) → Menos resultados, más precisos
   - Disminuir (ej: 0.20) → Más resultados, menos precisos
   - Usar 0.25-0.30 para balance

2. DYNAMIC_DROPOFF_RATIO (actualmente 0.65):
   - Aumentar (ej: 0.75) → Más estricto, descarta más candidatos débiles
   - Disminuir (ej: 0.55) → Más permisivo, retiene más candidatos
   - Usar 0.60-0.70 para balance

3. PROXIMITY_MAX_SCORE (actualmente 0.3):
   - Aumentar (ej: 0.4) → Da más peso a coincidencias numéricas exactas
   - Disminuir (ej: 0.2) → Da menos peso a coincidencias numéricas
   - Usar 0.2-0.4 dependiendo de importancia del dosage

4. SCORING_WEIGHTS:
   - semantic: 0.6, proximity: 0.4 → Balance actual
   - semantic: 0.7, proximity: 0.3 → Prioriza semántica (para queries vagas)
   - semantic: 0.5, proximity: 0.5 → Prioriza números (para queries específicas)

5. BOOST_WEIGHTS:
   - title: 0.6 → Coincidencia en nombre es muy importante
   - enterprise: 0.5 → Coincidencia en laboratorio es importante
   - Ajustar según lo que más valor tenga para los usuarios

6. FUZZY_MATCH_CUTOFF (actualmente 0.85):
   - Aumentar (ej: 0.90) → Solo typos muy similares
   - Disminuir (ej: 0.80) → Acepta más variaciones
   - Usar 0.80-0.90 para balance

EJEMPLO DE TUNING POR CASO DE USO:

Caso A: Usuarios con queries muy específicas ("apoquel 16mg")
→ SEMANTIC_THRESHOLD: 0.30 (más estricto)
→ PROXIMITY_MAX_SCORE: 0.4 (más peso a números)
→ DROPOFF_RATIO: 0.70 (más estricto)

Caso B: Usuarios con queries vagas ("algo para pulgas")
→ SEMANTIC_THRESHOLD: 0.20 (más permisivo)
→ PROXIMITY_MAX_SCORE: 0.2 (menos peso a números)
→ DROPOFF_RATIO: 0.60 (más permisivo)

Caso C: Balance general (actual)
→ SEMANTIC_THRESHOLD: 0.25
→ PROXIMITY_MAX_SCORE: 0.3
→ DROPOFF_RATIO: 0.65
"""