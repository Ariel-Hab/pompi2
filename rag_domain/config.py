"""
Configuración V4 - Parámetros para NER High Recall + LLM Judge
"""

# ═══════════════════════════════════════════════════════════════
# NER CLASSIFIER CONFIG
# ═══════════════════════════════════════════════════════════════

# Threshold de similitud para fuzzy matching (0-100)
# Valores más bajos = más candidatos (mayor recall)
NER_SIMILARITY_THRESHOLDS = {
    'PRODUCTO': 75,       # Bajo para capturar variantes
    'LABORATORIO': 80,
    'DROGA': 85,          # Alto para evitar falsos positivos químicos
    'CATEGORIA': 80,
    'ACCION': 80,
    'CONCEPTO': 80,
    'ESPECIE': 80
}

# Número máximo de candidatos a retornar por tipo
# (evita saturar el LLM con cientos de productos)
MAX_CANDIDATES_PER_TYPE = {
    'PRODUCTO': 50,
    'LABORATORIO': 10,
    'DROGA': 20,
    'CATEGORIA': 10,
    'ACCION': 15,
    'CONCEPTO': 10,
    'ESPECIE': 10
}

# Longitud de N-Gramas para búsqueda
# (1, 2, 3) = unigrams, bigrams, trigrams
NGRAM_RANGE = (1, 3)


# ═══════════════════════════════════════════════════════════════
# OPTIMIZER CONFIG
# ═══════════════════════════════════════════════════════════════

# Número de búsquedas históricas a incluir en el contexto del LLM
HISTORY_CONTEXT_LIMIT = 3

# Temperatura del LLM para el análisis (0.0 = determinístico)
LLM_JUDGE_TEMPERATURE = 0.0

# Longitud mínima de palabra para considerarla "raíz de familia"
MIN_FAMILY_ROOT_LENGTH = 3

# Máxima cantidad de palabras en una raíz de familia
# Ej: "POWER" = 1 palabra, "ROYAL CANIN" = 2 palabras
MAX_FAMILY_ROOT_WORDS = 2


# ═══════════════════════════════════════════════════════════════
# SEARCH CONFIG
# ═══════════════════════════════════════════════════════════════

# Pesos de scoring
SEARCH_WEIGHTS = {
    'semantic': 1.0,
    'keyword_fts': 2.0,
    'ner_similarity': 4.0,
    'family_match': 3.0,
    'dosage': 0.5
}

# Top-K por intención
TOP_K_BY_INTENT = {
    'SEARCH': 5,
    'RECOMMENDATION': 3,
    'SMALLTALK': 0
}

# Multiplicador de candidatos intermedios (antes de ranking final)
# Ej: top_k=5 * 10 = 50 candidatos iniciales
CANDIDATE_MULTIPLIER = 10

# Threshold mínimo de score total para incluir un resultado
MIN_TOTAL_SCORE = 0.1


# ═══════════════════════════════════════════════════════════════
# FAMILY CLUSTERING CONFIG
# ═══════════════════════════════════════════════════════════════

# Longitud máxima de nombre de producto para considerarlo "raíz"
# Ej: "POWER" (5 chars) vs "Power Gold 10kg" (16 chars)
MAX_FAMILY_ROOT_NAME_LENGTH = 15

# Número mínimo de SKUs para activar family clustering
# Si hay 3+ variantes de "Power", se colapsa a "POWER"
MIN_SKUS_FOR_CLUSTERING = 3

# Keywords que indican consulta genérica (activan family search)
GENERIC_QUERY_KEYWORDS = [
    'productos', 'linea', 'línea', 'catalogo', 'catálogo',
    'precios', 'que tiene', 'que tenes', 'ofertas de'
]


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def get_top_k_for_intent(intent: str) -> int:
    """Retorna top-k según intención"""
    return TOP_K_BY_INTENT.get(intent, TOP_K_BY_INTENT['SEARCH'])


def is_generic_query(query: str) -> bool:
    """Detecta si es consulta genérica por keywords"""
    query_lower = query.lower()
    return any(kw in query_lower for kw in GENERIC_QUERY_KEYWORDS)


def should_cluster_by_family(
    candidates_count: int,
    approved_count: int,
    query: str
) -> bool:
    """
    Decide si aplicar family clustering basándose en:
    1. Ratio candidatos/aprobados
    2. Número mínimo de SKUs
    3. Keywords genéricos en query
    """
    # Si el LLM filtró mucho, probablemente ya hizo clustering
    if approved_count < MIN_SKUS_FOR_CLUSTERING:
        return False
    
    # Si hay muchos candidatos y pocos aprobados, puede ser específico
    ratio = candidates_count / max(approved_count, 1)
    if ratio > 5:  # Ej: 50 candidatos / 2 aprobados = filtrado agresivo
        return False
    
    # Si la query es genérica, aplicar clustering
    if is_generic_query(query):
        return True
    
    return False


# ═══════════════════════════════════════════════════════════════
# DEVELOPMENT / DEBUG
# ═══════════════════════════════════════════════════════════════

# Activa logs verbosos
DEBUG_MODE = False

# Guarda candidatos del NER en archivo para análisis
SAVE_NER_CANDIDATES = False

# Path para guardar análisis (solo si SAVE_NER_CANDIDATES=True)
NER_ANALYSIS_PATH = "./debug/ner_candidates.json"


if __name__ == "__main__":
    # Test de configuración
    print("🔧 CONFIGURACIÓN V4 - HIGH RECALL + FAMILY CLUSTERING")
    print("="*60)
    
    print("\n📊 Thresholds NER:")
    for entity_type, threshold in NER_SIMILARITY_THRESHOLDS.items():
        print(f"  {entity_type}: {threshold}%")
    
    print(f"\n🔍 N-Gramas: {NGRAM_RANGE}")
    print(f"📜 Historial: {HISTORY_CONTEXT_LIMIT} búsquedas")
    
    print("\n⚖️ Pesos de Búsqueda:")
    for metric, weight in SEARCH_WEIGHTS.items():
        print(f"  {metric}: {weight}")
    
    print(f"\n📦 Top-K por Intent:")
    for intent, k in TOP_K_BY_INTENT.items():
        print(f"  {intent}: {k}")
    
    print(f"\n🌳 Family Clustering:")
    print(f"  Min SKUs: {MIN_SKUS_FOR_CLUSTERING}")
    print(f"  Max Root Length: {MAX_FAMILY_ROOT_NAME_LENGTH}")
    print(f"  Generic Keywords: {len(GENERIC_QUERY_KEYWORDS)}")
    
    # Test de helpers
    print("\n🧪 Tests:")
    test_queries = [
        "productos power",
        "power gold 10kg",
        "bravecto 20-40kg",
        "que tiene de brouwer"
    ]
    
    for query in test_queries:
        is_generic = is_generic_query(query)
        print(f"  '{query}' → Generic: {is_generic}")