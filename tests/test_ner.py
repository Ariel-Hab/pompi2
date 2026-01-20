import pytest
import logging
import sys
from rag_domain.ner_classifier import VeterinaryNERClassifier, classification_to_optimizer_format

## python -m pytest tests/test_ner.py -v -s

# ============================================================================
# ⚙️ CONFIGURACIÓN DE LOGGING
# ============================================================================
# Esto asegura que los logs se vean en consola al usar 'pytest -s'
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# ============================================================================
# ⚙️ FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def classifier():
    print("\n⚡ Inicializando VeterinaryNERClassifier...")
    return VeterinaryNERClassifier()

# ============================================================================
# 📋 DATOS DE PRUEBA
# ============================================================================

NER_LOGIC_CASES = [
    ("busco productos clinicos", ["CATEGORIA"]),
    ("necesito comprimido", ["CONCEPTO"]),
    ("antibiotico para perros", ["ACCION", "ESPECIE"]),
    ("meloxicam gato", ["DROGA", "ESPECIE"]),
    # Fuzzy / Typos
    ("atibiotico oral", ["ACCION"]),
    ("alimento holliday", ["CATEGORIA", "LABORATORIO"]),
    ("precio de sinparica", ["PRODUCTO"]),
    # Enriquecimiento
    ("pipeta para gatitos", ["CONCEPTO", "ESPECIE"]),
    ("vacuna cachorros", ["CONCEPTO", "ESPECIE"]),
    # Complejidad
    ("bravecto para perros", ["PRODUCTO", "ESPECIE"]),
    ("antiparasitarios jhon martin para gatos", ["CATEGORIA", "LABORATORIO", "ESPECIE"]),
    ("collares para pulgas", ["CONCEPTO", "ESPECIE"]),
]

INTEGRATED_SEARCH_CASES = [
    ("fenobarbital 40 mg", {
        "types_in_result": ["DROGA"], 
        "filters": {"dosage_value": 40.0, "dosage_unit": "mg"}
    }),
    ("anticonvulsivante 40mg", {
        "types_in_result": ["ACCION"], 
        "filters": {"dosage_value": 40.0, "dosage_unit": "mg"}
    }),
    ("multivit boehringer", {
        "types_in_result": ["LABORATORIO"],
    }),
    ("total full gatos comprimidos", {
        "types_in_result": ["PRODUCTO", "ESPECIE", "CONCEPTO"],
        "filters": {"presentation": "comprimidos"}
    }),
    ("power gatos 5kg", {
        "types_in_result": ["PRODUCTO", "ESPECIE"], 
    }),
    ("basken hospitalario", {
        "types_in_result": ["PRODUCTO"],
    }),
]

INTEGRATED_SEARCH_CASES += [
    # ESCENARIO: Seguridad Gatos (Duosecretina)
    # Aquí veremos si el NER identifica que es para perros o si ensucia la búsqueda
    ("duosecretina para gatos", {
        "types_in_result": ["PRODUCTO", "ESPECIE"],
        "filters": {"species_filter": ["gato"]} # Esto es lo que causó el error: el filtro existía pero no bloqueó el resultado
    }),
    # ESCENARIO: Búsqueda por Acción Clínica (Osteoartritis)
    # Verificamos si el NER extrae la patología como ACCION
    ("algo para la osteoartritis en perros", {
        "types_in_result": ["ACCION", "ESPECIE"],
        "filters": {"species_filter": ["perro"]}
    }),
    # ESCENARIO: Laboratorio + Categoría (Ceva Clínico)
    # El bot devolvió Doxivit, veremos si el NER detectó ambos filtros
    ("productos ceva linea clinica", {
        "types_in_result": ["LABORATORIO", "CATEGORIA"],
        "filters": {"filter_lab": "ceva", "filter_category": "clinico"}
    })
]

# ============================================================================
# 🧪 TESTS DE CLASIFICACIÓN (LÓGICA NER)
# ============================================================================

@pytest.mark.parametrize("query, expected_types", NER_LOGIC_CASES)
def test_ner_entity_detection(classifier, query, expected_types):
    result = classifier.classify(query)
    found_types = [e.entity_type for e in result.all_entities]
    
    # --- LOGGING DETALLADO ---
    logger.info(f"\n{'='*70}")
    logger.info(f"🧪 TEST QUERY: '{query}'")
    logger.info(f"{'-'*70}")
    logger.info(f"🎯 Tipos Esperados: {expected_types}")
    logger.info(f"🤖 Tipos Detectados: {found_types}")
    
    # Detalle de qué texto matcheó con qué tipo
    details = [f"[{e.entity_type}: '{e.entity_value}']" for e in result.all_entities]
    logger.info(f"📝 Entidades:      {' '.join(details)}")
    
    missing = [t for t in expected_types if t not in found_types]
    
    # Marcador visual de éxito/fracaso en el log antes del assert
    if missing:
        logger.info(f"❌ RESULTADO: FALLO (Faltan: {missing})")
    else:
        logger.info(f"✅ RESULTADO: OK")
    logger.info(f"{'='*70}")

    # --- ASSERT ---
    assert not missing, (
        f"Faltan tipos esperados en '{query}': {missing}. "
        f"Encontrado: {found_types}"
    )

# ============================================================================
# 🧪 TESTS DE INTEGRACIÓN (FILTROS)
# ============================================================================

@pytest.mark.parametrize("query, expectations", INTEGRATED_SEARCH_CASES)
def test_ner_integration_data(classifier, query, expectations):
    result = classifier.classify(query)
    found_types = [e.entity_type for e in result.all_entities]

    # --- LOGGING DETALLADO ---
    logger.info(f"\n{'='*70}")
    logger.info(f"⚙️  INTEGRATION TEST: '{query}'")
    logger.info(f"{'-'*70}")
    
    # Loguear entidades encontradas
    details = [f"[{e.entity_type}: '{e.entity_value}']" for e in result.all_entities]
    logger.info(f"📝 Entidades: {details}")
    
    # Loguear filtros encontrados vs esperados
    logger.info(f"🔍 Filtros Detectados: {result.filters}")
    if "filters" in expectations:
        logger.info(f"🎯 Filtros Esperados:  {expectations['filters']}")
    
    logger.info(f"{'='*70}")

    # 1. Validar Tipos
    if "types_in_result" in expectations:
        for t in expectations["types_in_result"]:
            assert t in found_types, f"Falta el tipo {t}"

    # 2. Validar Filtros
    if "filters" in expectations:
        for key, value in expectations["filters"].items():
            assert key in result.filters, f"Falta filtro '{key}'"
            assert result.filters[key] == value, f"Valor erróneo en '{key}'"

# ============================================================================
# 🧪 OTROS TESTS
# ============================================================================

def test_classification_to_optimizer_format(classifier):
    query = "antiparasitario bayer para gatos 5kg"
    classification = classifier.classify(query)
    optimizer_payload = classification_to_optimizer_format(classification)
    
    logger.info(f"\n📦 PAYLOAD OPTIMIZER GENERADO:\n{optimizer_payload}")
    
    assert "intent" in optimizer_payload
    assert "main_entity" in optimizer_payload
    assert "details" in optimizer_payload
    assert "parsed_metadata" in optimizer_payload

def test_intent_detection(classifier):
    res_st = classifier.classify("hola buenos dias")
    logger.info(f"\n🗣️ Intent Test 1: 'hola buenos dias' -> {res_st.intent}")
    assert res_st.intent == "SMALLTALK"

    res_mix = classifier.classify("hola tenes bravecto")
    logger.info(f"🗣️ Intent Test 2: 'hola tenes bravecto' -> {res_mix.intent}")
    assert res_mix.intent == "SEARCH"