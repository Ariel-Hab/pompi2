import pytest
import json
from chat_domain.manager import ConversationManager

# python -m pytest tests/test_manager.py -v -s

# Configuración de Colores para Logs
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

@pytest.fixture(scope="function")
def manager(request):
    """
    Inicializa el Manager en Modo Test para cada test individual.
    """
    query = getattr(request, 'param', 'default_test')
    session_id = f"test_session_{hash(query)}"
    
    print(f"\n{Colors.HEADER}🔌 [INIT] Inicializando Manager para: '{query[:50]}...'{Colors.ENDC}")
    
    mgr = ConversationManager(
        session_id=session_id, 
        user_id="tester_01"
    )
    mgr.test_mode = True
    return mgr


@pytest.mark.parametrize("query, expected_intent, expected_context_type", [
    
    # --- ESCENARIOS ORIGINALES ---
    ("Busco productos de laboratorio CEVA que sean de la línea clínica", "SEARCH", "product"),
    ("¿Puedo usar Duosecretina en mi gato para problemas hepáticos?", "RECOMMENDATION", "product"),
    ("Necesito algo que tenga Gabapentina para un perro con dolor neuropático", "SEARCH", "product"),
    ("¿Cuántos comprimidos de Duosecretina le doy a un perro de 20kg?", "RECOMMENDATION", "product"),
    ("Tengo un perro de 15kg con pulgas, ¿cuál de los Zanex le corresponde?", "SEARCH", "product"),
    ("¿Tienen algún anticonceptivo inyectable para perras?", "SEARCH", "product"),
    ("¿Qué me recomiendas para un perro con signos de osteoartritis?", "RECOMMENDATION", "product"),
    ("¿Qué contraindicaciones tiene el Neo Vitapel de Brouwer?", "SEARCH", "product"),
    ("Busco el alimento de Holliday para problemas cardíacos de 10kg", "SEARCH", "product"),

    # --- NUEVOS ESCENARIOS BASADOS EN EMBEDDINGS CARGADOS ---

    # 1. NUEVO LANZAMIENTO (Pregabaliv - John Martin)
    # Prueba detección de "nuevo producto" y droga específica
    ("Quiero información sobre el nuevo analgésico de John Martin con Pregabalina", "SEARCH", "product"),

    # 2. OFERTA ESPECÍFICA CON REGALO (Labyderm - Labyes)
    # Prueba búsqueda de promociones complejas (regalo/riñonera)
    ("Busco la oferta de Labyderm que viene con una riñonera de regalo", "SEARCH", "offer"),

    # 3. TRANSFER / BONIFICACIÓN (Holliday)
    # Prueba búsqueda de reglas de transfer comerciales
    ("¿Qué promociones vigentes o transfers tiene el laboratorio Holliday?", "SEARCH", "transfer"),

    # 4. BÚSQUEDA POR PESO Y ESPECIE (Credelio - Elanco)
    # Prueba filtro preciso de peso (2.5kg) y especie
    ("Necesito una pastilla para pulgas para un perro muy chiquito de 2.5kg", "RECOMMENDATION", "product"),

    # 5. PREGUNTA TÉCNICA/CLÍNICA (Pets Protector - Ceva)
    # Prueba recuperación de info técnica (droga/acción)
    ("¿Qué droga tiene el Pets Protector Max y para qué sirve?", "SEARCH", "product"),
    
    # 6. TRANSFER ESPECÍFICO (Konig)
    # Prueba búsqueda directa de un transfer por nombre comercial
    ("Mostrame el transfer de Dominal Max", "SEARCH", "transfer")
])
def test_manager_pipeline_visual(query, expected_intent, expected_context_type):
    """
    Ejecuta el pipeline completo y visualiza:
    1. NER
    2. Enriquecimiento del Contexto (Vademecum)
    3. Respuesta LLM
    4. Attachments Finales (Productos)
    """
    
    print(f"\n\n{Colors.BOLD}{'='*100}{Colors.ENDC}")
    print(f"🧪 {Colors.CYAN}ESCENARIO: {query}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*100}{Colors.ENDC}")

    # 1. Inicialización del Manager DENTRO del test
    session_id = f"test_session_{hash(query)}"
    manager = ConversationManager(
        session_id=session_id,
        user_id="tester_01"
    )
    manager.test_mode = True
    
    # 2. Ejecución
    response = manager.handle_message(query, generate_response=True)
    
    # 3. Verificación de que se guardó el resultado
    assert manager.last_test_result is not None, \
        f"❌ Test result no fue guardado para query: '{query}'"
    
    debug_data = manager.last_test_result
    
    # 4. Visualización NER
    print(f"\n🎯 {Colors.BOLD}[PASO 1] NER & CLASIFICACIÓN{Colors.ENDC}")
    print(f"   Intent Detectado: {Colors.BLUE}{debug_data.ner_intent}{Colors.ENDC}")
    print(f"   Entidad Principal: {debug_data.ner_primary_entity} ({debug_data.ner_entity_type})")
    print(f"   Filtros Aplicados: {debug_data.ner_filters}")
    
    if debug_data.ner_intent != expected_intent and expected_intent != "SMALLTALK":
        print(f"   {Colors.WARNING}⚠️  Advertencia: Se esperaba {expected_intent}{Colors.ENDC}")

    # 5. Verificación de resultados
    if not debug_data.has_results and expected_context_type:
        print(f"\n❌ {Colors.FAIL}FAIL: No se encontraron resultados en el Vector Search.{Colors.ENDC}")
        print(f"   Query Optimizada: {debug_data.optimized_query}")
    
    # 6. Visualización ENRIQUECIMIENTO (Lo que ve el LLM)
    if debug_data.search_results:
        print(f"\n🧠 {Colors.BOLD}[PASO 2] CONTEXTO PARA LLM (RAG){Colors.ENDC}")
        print(f"   Candidatos recuperados: {len(debug_data.search_results)}")
        
        # Inspeccionamos los top 3 candidatos
        for idx, result in enumerate(debug_data.search_results[:3], 1):
            top_candidate = result.get('metadata', {})
            scores = result.get('scores', {})
            
            print(f"\n   🔎 {Colors.BOLD}Candidato #{idx}:{Colors.ENDC}")
            
            # Nombre del producto/oferta/transfer
            nombre = (
                top_candidate.get('PRODUCTO') or 
                top_candidate.get('product_name') or 
                top_candidate.get('title', 'N/A')
            )
            
            # Laboratorio
            lab = (
                top_candidate.get('LABORATORIO') or
                top_candidate.get('laboratorio') or 
                top_candidate.get('supplier') or 
                top_candidate.get('enterprise_title', 'N/A')
            )
            
            print(f"      Nombre: {Colors.CYAN}{nombre}{Colors.ENDC}")
            print(f"      Laboratorio: {lab}")
            print(f"      Tipo: {top_candidate.get('type', 'unknown')}")
            
            # Scores
            print(f"      📊 Scores: Total={scores.get('total', 0):.4f} | "
                  f"Sem={scores.get('semantic', 0):.2f} | "
                  f"Key={scores.get('keyword', 0):.2f} | "
                  f"NER={scores.get('ner', 0):.2f}")
            
            # Datos específicos para debugging visual
            desc = top_candidate.get('description', '')
            if desc:
                 print(f"      Desc: {desc[:100]}...")

    else:
        print(f"\n{Colors.WARNING}⚠️  No se recuperaron candidatos del Vector Search{Colors.ENDC}")

    # 7. Visualización RESPUESTA LLM
    print(f"\n🤖 {Colors.BOLD}[PASO 3] RESPUESTA GENERADA{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-'*80}{Colors.ENDC}")
    print(debug_data.final_response.strip())
    print(f"{Colors.BLUE}{'-'*80}{Colors.ENDC}")

    # 8. VEREDICTO VISUAL
    print(f"\n{'='*100}")
    assert debug_data.ner_intent is not None, "Intent no fue detectado"
    assert debug_data.final_response is not None, "No se generó respuesta"
    print(f"✅ {Colors.GREEN}{Colors.BOLD}TEST COMPLETADO{Colors.ENDC}")
    print(f"{'='*100}\n")


def test_conversation_multiturn():
    """
    Test de Conversación (Memoria y Contexto).
    Simula un flujo de 3 mensajes para verificar que el bot mantiene el contexto
    (ej: recuerda que hablamos de un perro de cierto peso).
    """
    print(f"\n\n{Colors.HEADER}🗣️  TEST DE CONVERSACIÓN MULTI-TURN (MEMORIA){Colors.ENDC}")
    
    # Usamos un session_id fijo para simular la misma sesión
    session_id = "test_conversation_session_001"
    manager = ConversationManager(session_id=session_id, user_id="tester_conv")
    manager.test_mode = True

    # --- TURNO 1: Búsqueda General ---
    q1 = "Hola, busco antiparasitarios para perros"
    print(f"\n{Colors.BOLD}👤 Usuario (1):{Colors.ENDC} {q1}")
    
    resp1 = manager.handle_message(q1, generate_response=True)
    print(f"{Colors.BLUE}🤖 Bot (1):{Colors.ENDC} {manager.last_test_result.final_response[:150]}...")
    
    # Verificamos que entendió la categoría
    assert "antiparasitario" in str(manager.last_test_result.ner_filters).lower() or \
           "antiparasitario" in str(manager.last_test_result.ner_primary_entity).lower()

    # --- TURNO 2: Refinamiento por Peso (Contexto) ---
    q2 = "Es para uno chiquito, de 2 kilos"
    print(f"\n{Colors.BOLD}👤 Usuario (2):{Colors.ENDC} {q2}")
    
    resp2 = manager.handle_message(q2, generate_response=True)
    print(f"{Colors.BLUE}🤖 Bot (2):{Colors.ENDC} {manager.last_test_result.final_response[:150]}...")
    
    # Debería haber encontrado productos como CREDELIO o PETS PROTECTOR MAX (rango bajo)
    results_t2 = manager.last_test_result.search_results
    found_relevant = False
    for res in results_t2:
        name = res['metadata'].get('title', '').upper()
        # Verificamos si trajo Credelio o Pets Protector que son para < 2.5/5kg
        if "CREDELIO" in name or "PETS PROTECTOR" in name:
            found_relevant = True
            print(f"   ✅ Contexto aplicado, encontró: {name}")
            break
    
    if not found_relevant:
        print(f"   {Colors.WARNING}⚠️  Advertencia: No parece haber filtrado por peso en el Turno 2{Colors.ENDC}")

    # --- TURNO 3: Pregunta Específica sobre resultado previo ---
    q3 = "¿Qué droga tiene el Credelio?"
    print(f"\n{Colors.BOLD}👤 Usuario (3):{Colors.ENDC} {q3}")
    
    resp3 = manager.handle_message(q3, generate_response=True)
    print(f"{Colors.BLUE}🤖 Bot (3):{Colors.ENDC} {manager.last_test_result.final_response}")

    # Verificamos que responda sobre Lotilaner
    response_text = manager.last_test_result.final_response.lower()
    if "lotilaner" in response_text:
        print(f"   ✅ Respuesta correcta (Menciona Lotilaner)")
    else:
        print(f"   {Colors.FAIL}❌ Respuesta incorrecta (No menciona Lotilaner){Colors.ENDC}")

    print(f"\n{Colors.GREEN}✅ TEST DE CONVERSACIÓN FINALIZADO{Colors.ENDC}")