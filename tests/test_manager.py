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

@pytest.fixture(scope="function")  # CAMBIADO: function en lugar de module
def manager(request):
    """
    Inicializa el Manager en Modo Test para cada test individual.
    Usa el query como parte del session_id para evitar colisiones.
    """
    # Obtenemos el parámetro del test actual si existe
    query = getattr(request, 'param', 'default_test')
    session_id = f"test_session_{hash(query)}"
    
    print(f"\n{Colors.HEADER}🔌 [INIT] Inicializando Manager para: '{query[:50]}...'{Colors.ENDC}")
    
    mgr = ConversationManager(
        session_id=session_id, 
        user_id="tester_01"
    )
    mgr.test_mode = True  # Habilitamos la captura de datos internos
    
    return mgr


@pytest.mark.parametrize("query, expected_intent, expected_context_type", [
    
    # --- ESCENARIO 1: FILTRO POR LABORATORIO Y CATEGORÍA ---
    ("Busco productos de laboratorio CEVA que sean de la línea clínica", "SEARCH", "product"),
    
    # --- ESCENARIO 2: RESTRICCIÓN POR ESPECIE (PROTECCIÓN) ---
    ("¿Puedo usar Duosecretina en mi gato para problemas hepáticos?", "RECOMMENDATION", "product"),
    
    # --- ESCENARIO 3: BÚSQUEDA POR COMPONENTE (DROGA) ---
    ("Necesito algo que tenga Gabapentina para un perro con dolor neuropático", "SEARCH", "product"),
    
    # --- ESCENARIO 4: DOSIFICACIÓN BASADA EN PESO ---
    ("¿Cuántos comprimidos de Duosecretina le doy a un perro de 20kg?", "RECOMMENDATION", "product"),
    
    # --- ESCENARIO 5: DIFERENCIACIÓN DE PRESENTACIONES (PESO ESPECÍFICO) ---
    ("Tengo un perro de 15kg con pulgas, ¿cuál de los Zanex le corresponde?", "SEARCH", "product"),
    
    # --- ESCENARIO 6: BÚSQUEDA POR ACCIÓN TERAPÉUTICA ---
    ("¿Tienen algún anticonceptivo inyectable para perras?", "SEARCH", "product"),
    
    # --- ESCENARIO 7: INDICACIONES CLÍNICAS ESPECÍFICAS ---
    ("¿Qué me recomiendas para un perro con signos de osteoartritis?", "RECOMMENDATION", "product"),
    
    # --- ESCENARIO 8: COMPLIANCE Y SEGURIDAD ---
    ("¿Qué contraindicaciones tiene el Neo Vitapel de Brouwer?", "SEARCH", "product"),
    
    # --- ESCENARIO 9: PRODUCTOS DE ALIMENTACIÓN ESPECÍFICA ---
    ("Busco el alimento de Holliday para problemas cardíacos de 10kg", "SEARCH", "product")
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
    print(f"   Todas las Entidades: {debug_data.ner_all_entities}")
    print(f"   Filtros Aplicados: {debug_data.ner_filters}")
    
    if debug_data.ner_intent != expected_intent and expected_intent != "SMALLTALK":
        print(f"   {Colors.WARNING}⚠️  Advertencia: Se esperaba {expected_intent}{Colors.ENDC}")

    # 5. Verificación de resultados
    if not debug_data.has_results and expected_context_type:
        print(f"\n❌ {Colors.FAIL}FAIL: No se encontraron resultados en el Vector Search.{Colors.ENDC}")
        print(f"   Query Optimizada: {debug_data.optimized_query}")
        # No retornamos, continuamos para ver la respuesta del LLM
    
    # 6. Visualización ENRIQUECIMIENTO (Lo que ve el LLM)
    if debug_data.search_results:
        print(f"\n🧠 {Colors.BOLD}[PASO 2] CONTEXTO PARA LLM (RAG){Colors.ENDC}")
        print(f"   Candidatos recuperados: {len(debug_data.search_results)}")
        
        # Inspeccionamos los top 3 candidatos
        for idx, result in enumerate(debug_data.search_results[:3], 1):
            top_candidate = result.get('metadata', {})
            scores = result.get('scores', {})
            
            print(f"\n   🔎 {Colors.BOLD}Candidato #{idx}:{Colors.ENDC}")
            
            # Nombre del producto
            nombre = (
                top_candidate.get('PRODUCTO') or 
                top_candidate.get('product_name') or 
                top_candidate.get('title', 'N/A')
            )
            
            # Laboratorio
            lab = (
                top_candidate.get('LABORATORIO') or
                top_candidate.get('laboratorio') or 
                top_candidate.get('enterprise_title', 'N/A')
            )
            
            print(f"      Nombre: {Colors.CYAN}{nombre}{Colors.ENDC}")
            print(f"      Laboratorio: {lab}")
            
            # Scores
            print(f"      📊 Scores: Total={scores.get('total', 0):.4f} | "
                  f"Sem={scores.get('semantic', 0):.2f} | "
                  f"Key={scores.get('keyword', 0):.2f} | "
                  f"NER={scores.get('ner', 0):.2f}")
            
            # Datos específicos según tipo
            if expected_context_type == 'product':
                # Información clínica
                accion = (
                    top_candidate.get('ACCION TERAPEUTICA') or 
                    top_candidate.get('therapeutic_action') or
                    top_candidate.get('description', '')
                )
                
                droga = (
                    top_candidate.get('DROGA') or
                    top_candidate.get('active_ingredient', '')
                )
                
                presentacion = (
                    top_candidate.get('CONCEPTO') or
                    top_candidate.get('presentacion', '')
                )
                
                if accion:
                    print(f"      Acción Terapéutica: {Colors.GREEN}{accion[:80]}...{Colors.ENDC}")
                if droga:
                    print(f"      Principio Activo: {droga}")
                if presentacion:
                    print(f"      Presentación: {presentacion}")
                    
            elif expected_context_type == 'offer':
                desc = top_candidate.get('description', 'N/A')
                print(f"      Descripción: {Colors.CYAN}{desc[:80]}...{Colors.ENDC}")
    else:
        print(f"\n{Colors.WARNING}⚠️  No se recuperaron candidatos del Vector Search{Colors.ENDC}")

    # 7. Visualización RESPUESTA LLM
    print(f"\n🤖 {Colors.BOLD}[PASO 3] RESPUESTA GENERADA{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-'*80}{Colors.ENDC}")
    print(debug_data.final_response.strip())
    print(f"{Colors.BLUE}{'-'*80}{Colors.ENDC}")

    # 8. Visualización ATTACHMENTS (Lo que ve el Frontend)
    attachments = response.get('attachments', [])
    print(f"\n📦 {Colors.BOLD}[PASO 4] ATTACHMENTS FRONTEND (Tarjetas){Colors.ENDC}")
    
    if attachments:
        print(f"   Se generaron {len(attachments)} tarjetas visuales.")
        
        for idx, att in enumerate(attachments[:3], 1):
            att_data = att.get('data', {})
            
            print(f"\n   🔎 {Colors.BOLD}Tarjeta #{idx}:{Colors.ENDC}")
            print(f"      Título: {att_data.get('title', 'N/A')}")
            
            # Verificamos datos comerciales si aplica
            if expected_context_type == 'product':
                price = att_data.get('selling_price', att_data.get('list_price', 'N/A'))
                has_offer = att_data.get('has_offer', False)
                
                print(f"      Precio: {Colors.GREEN}${price}{Colors.ENDC}")
                print(f"      En Oferta: {has_offer}")
                
            elif expected_context_type == 'offer':
                discount = att_data.get('cash_discount_percentaje', 'N/A')
                print(f"      Descuento: {Colors.GREEN}{discount}%{Colors.ENDC}")
    else:
        if expected_context_type:
            print(f"   {Colors.WARNING}⚠️  Sin attachments (¿Es correcto para esta query?){Colors.ENDC}")
        else:
            print(f"   ✓ (Correcto: Esta query no requiere attachments)")

    # 9. VEREDICTO VISUAL
    print(f"\n{'='*100}")
    
    # Assertions opcionales (puedes comentarlas si solo quieres ver el output)
    assert debug_data.ner_intent is not None, "Intent no fue detectado"
    assert debug_data.final_response is not None, "No se generó respuesta"
    
    print(f"✅ {Colors.GREEN}{Colors.BOLD}TEST COMPLETADO{Colors.ENDC}")
    print(f"{'='*100}\n")


# Test adicional para verificar el flujo sin parametrización
def test_single_query_debug():
    """Test individual para debugging detallado"""
    
    query = "Busco productos de laboratorio CEVA"
    
    print(f"\n{Colors.HEADER}🔬 DEBUG TEST: '{query}'{Colors.ENDC}")
    
    manager = ConversationManager(
        session_id="debug_session",
        user_id="debug_user"
    )
    manager.test_mode = True
    
    print(f"{Colors.CYAN}Test mode activado: {manager.test_mode}{Colors.ENDC}")
    
    response = manager.handle_message(query, generate_response=True)
    
    print(f"{Colors.CYAN}Respuesta recibida: {response is not None}{Colors.ENDC}")
    print(f"{Colors.CYAN}Last test result: {manager.last_test_result is not None}{Colors.ENDC}")
    
    if manager.last_test_result:
        manager.last_test_result.print_summary()
    else:
        print(f"{Colors.FAIL}❌ NO SE GUARDÓ EL TEST RESULT{Colors.ENDC}")
        print(f"   Verificar que _save_test_result se está llamando")
        print(f"   Response keys: {response.keys()}")