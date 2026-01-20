import pytest
import json
import logging
import sys
from rag_domain.optimizer import QueryOptimizer
from rag_domain.search import VectorSearchService

## python -m pytest tests/integrated_search_test.py -v -s

# ============================================================================
# ⚙️ CONFIGURACIÓN DE LOGGING & FIXTURES
# ============================================================================

# Configuramos logging para que se vea en la consola al usar pytest -s
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s', # Formato limpio para no ensuciar la tabla
    stream=sys.stdout,
    force=True
)

@pytest.fixture(scope="module")
def services():
    return {
        "optimizer": QueryOptimizer(),
        "search": VectorSearchService()
    }

# ============================================================================
# 📋 DATOS DE PRUEBA
# ============================================================================
TEST_CASES = [
    # --- GRUPO A: Búsqueda Exacta (Droga + Dosis) ---
    ("fenobarbital 40 mg", 5827, "Droga + Dosis exacta"),
    ("anticonvulsivante 40mg", 5827, "Acción + Dosis"),

    # --- GRUPO B: Búsqueda por Laboratorio (Soft Boost -> Hard Filter) ---
    ("multivit boehringer", 5828, "Nombre + Lab"),
    ("shampoo osspret", 5835, "Categoría + Lab"),

    # --- GRUPO C: Formas Farmacéuticas ---
    ("total full gatos comprimidos", 5829, "Producto + Forma (Comprimidos)"),
    ("total full gatos suspension", 5837, "Producto + Forma (Suspensión)"),

    # --- GRUPO D: Rangos de Peso ---
    ("power gatos 5kg", 5830, "Producto + Peso (Límite Inf)"),
    ("ectholaner mas de 40 kg", 14147, "Producto + Peso Alto"),

    # --- GRUPO E: Etiquetas Especiales ---
    ("basken hospitalario", 5834, "Producto + Tag Hospitalario"),
]
TEST_CASES += [
    # ESCENARIO CRÍTICO: El bot falló aquí recomendando "Seraquin" (Condroprotector)
    # Queremos ver si DUOSECRETINA (ID 20360) aparece o si la semántica de "gato" lo aleja
    ("duosecretina problemas hepaticos gato", 20360, "Seguridad: Producto Prohibido en Gatos"),

    # ESCENARIO: El bot recomendó "Artrosan Equino" para perros
    # Queremos ver si RIMADYL (ID 20397) le gana en score a los productos de Equinos
    ("remedio para osteoartritis perros", 20397, "Lógica: Patología específica Perros"),

    # ESCENARIO: El bot no encontró el alimento cardiaco
    # Verificamos si el ID 20399 (MV Cardiaco) tiene buen score de proximidad
    ("alimento holliday cardiaco 10kg", 20399, "Categoría: Alimento medicado"),
    
    # ESCENARIO: El bot no encontró Neo Vitapel (ID 20400)
    ("contraindicaciones neo vitapel brouwer", 20400, "Nombre: Producto específico Brouwer")
]
# ============================================================================
# 🔬 TEST DE INSPECCIÓN VISUAL
# ============================================================================
@pytest.mark.parametrize("user_query, expected_id, scenario", TEST_CASES)
def test_visual_inspection(services, user_query, expected_id, scenario):
    optimizer = services["optimizer"]
    search_service = services["search"]

    print(f"\n\n{'='*100}")
    print(f"🧪 ESCENARIO: {scenario}")
    print(f"🗣️  Query Usuario: '{user_query}'")
    print(f"🎯 ID Esperado:   {expected_id}")
    print(f"{'-'*100}")

    # 1. OPTIMIZACIÓN
    optimized_req = optimizer.optimize(user_query)
    
    # Imprimir interpretación del optimizador
    print(f"🤖 OPTIMIZADOR (Interpretación):")
    filters = optimized_req.get('search_filters', {})
    if filters:
        print(json.dumps(filters, indent=2, ensure_ascii=False))
    else:
        print("   (Sin filtros detectados)")
        
    print(f"{'-'*100}")

    # 2. BÚSQUEDA (Los logs de search.py aparecerán aquí gracias al logging.basicConfig)
    results = search_service.search_with_context(optimized_req, top_k=5)

    # 3. REPORTE DE RESULTADOS
    if not results:
        print("❌ NO SE ENCONTRARON RESULTADOS.")
        return

    print(f"\n📚 TABLA RESUMEN (Top {len(results)}):")
    # Cabecera actualizada para V2.1 (Lab/Cat son filtros duros, no puntaje)
    header = f"{'#':<3} | {'ID':<6} | {'TOTAL':<7} | {'SEM':<6} {'KEY':<6} {'PROX':<6} | {'TÍTULO'}"
    print(header)
    print("-" * len(header))

    found_target = False

    for i, res in enumerate(results):
        _id = res['entity_id']
        total = res['total_score']
        metadata = res['metadata']
        title = metadata.get('title', 'Sin Título')[:45] 
        
        # Desglose de puntos (Mapeado a las claves nuevas de search.py)
        scores = res.get('_debug', {})
        sem = f"{scores.get('sem', 0):.4f}"
        key = f"{scores.get('key', 0):.1f}"
        prox = f"{scores.get('prox', 0):.4f}"

        # Marcador visual
        marker = ""
        if _id == expected_id:
            marker = "👈 🎯"
            found_target = True
        
        # Imprimir fila
        print(f"{i+1:<3} | {_id:<6} | {total:.3f}   | {sem:<6} {key:<6} {prox:<6} | {title} {marker}")

    print(f"{'='*100}")
    
    if not found_target:
        print(f"⚠️  FAIL: El ID esperado ({expected_id}) NO apareció en el Top 5.")
        # Opcional: Fallar el test formalmente si quieres automatización estricta
        # assert found_target, f"El ID {expected_id} no fue encontrado para '{user_query}'"