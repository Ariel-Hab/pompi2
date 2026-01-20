"""
CONVERSATION MANAGER V3 - Con Tests de LLM y Visualización de Objetos
"""
import traceback
import json
import sys
from typing import Dict, Any, List, Optional
from chat_domain.chat_history import ChatHistoryService
from rag_domain.ner_classifier import VeterinaryNERClassifier, classification_to_optimizer_format
from rag_domain.optimizer import QueryOptimizer
from rag_domain.prompts import build_rag_context, get_conversation_system_prompt
from ai_gateway.llm_client import LLMService
from chat_domain.attachment_builder import AttachmentBuilder
from rag_domain.config import get_top_k_for_intent
from rag_domain.search import VectorSearchService
from dataclasses import dataclass, asdict

# # Para ver output detallado
# python -m pytest tests/test_manager.py::test_manager_pipeline_visual -v -s

# # Para ejecutar solo un caso específico
# python -m pytest tests/test_manager.py::test_manager_pipeline_visual[Busco productos de laboratorio CEVA que sean de la línea clínica-SEARCH-product] -v -s

# # Para el test de debug
# python -m pytest tests/test_manager.py::test_single_query_debug -v -s

# ============================================================================
# DATA CLASSES PARA TESTING
# ============================================================================

@dataclass
class TestResult:
    """Resultado de un test de búsqueda"""
    query: str
    
    # NER Detection
    ner_intent: str
    ner_primary_entity: Optional[str]
    ner_entity_type: Optional[str]
    ner_all_entities: List[Dict]
    ner_filters: Dict
    ner_confidence: float
    
    # Search Results
    search_candidates_count: int
    search_results: List[Dict]
    
    # Final Output
    final_response: str
    final_attachments_count: int
    
    # Metadata
    has_results: bool
    optimized_query: Any = None 
    search_context: Optional[List] = None
    error: Optional[str] = None
    
    def to_dict(self):
        """Convierte a dict para fácil visualización"""
        return asdict(self)
    
    def print_summary(self):
        """Imprime resumen detallado incluyendo objetos y respuesta final"""
        print(f"\n{'='*80}")
        print(f"🧐 ANÁLISIS DEL TEST: '{self.query}'")
        print(f"{'='*80}")
        
        # 1. NER
        print(f"\n🎯 1. DETECCIÓN DE INTENCIÓN (NER):")
        print(f"   Intent: {self.ner_intent}")
        print(f"   Entidad Principal: {self.ner_entity_type} = {self.ner_primary_entity}")
        
        if isinstance(self.optimized_query, dict):
            print(f"   Query Optimizada: {self.optimized_query.get('main_entity')}")
        
        print(f"   Filtros Detectados: {self.ner_filters}")
        
        # 2. SEARCH OBJECTS
        print(f"\n📦 2. OBJETOS RECUPERADOS ({len(self.search_results)}):")
        
        if self.search_results:
            for idx, result in enumerate(self.search_results, 1):
                meta = result.get('metadata', {})
                scores = result.get('scores', {})
                
                # Detección flexible de tipo
                is_product = 'PRODUCTO' in meta or 'product_name' in meta
                
                print(f"   {'─'*60}")
                if is_product:
                    # Formato para PRODUCTOS
                    nombre = meta.get('PRODUCTO') or meta.get('product_name', 'N/A')
                    lab = meta.get('LABORATORIO') or meta.get('laboratorio', 'N/A')
                    precio = meta.get('PR LISTA', meta.get('list_price', 'N/A'))
                    accion = meta.get('ACCION TERAPEUTICA', meta.get('description', ''))
                    presentacion = meta.get('CONCEPTO', meta.get('presentacion', ''))
                    
                    print(f"   #{idx} [PRODUCTO] {nombre}")
                    print(f"       Lab: {lab} | Presentación: {presentacion}")
                    if accion:
                        print(f"       Acción: {accion[:100]}...")
                else:
                    # Formato para OFERTAS
                    title = meta.get('title', 'N/A')
                    desc = meta.get('description', '')
                    
                    print(f"   #{idx} [OFERTA] {title}")
                    if desc:
                        print(f"       Desc: {desc[:100]}...")

                # Scores técnicos
                print(f"       📊 Scores: Total={scores.get('total', 0.0):.4f} | "
                    f"Sem={scores.get('semantic', 0.0):.2f} | "
                    f"Key={scores.get('keyword', 0.0):.2f} | "
                    f"NER={scores.get('ner', 0.0):.2f}")
        else:
            print("   (Sin resultados en la base de datos)")

        # 3. FINAL LLM RESPONSE
        print(f"\n💬 3. RESPUESTA FINAL DEL LLM:")
        print(f"{'─'*80}")
        print(self.final_response)
        print(f"{'─'*80}")
        
        print(f"\n📎 Attachments generados: {self.final_attachments_count}")
        
        if self.error:
            print(f"\n❌ ERROR: {self.error}")


# ============================================================================
# CONVERSATION MANAGER V2
# ============================================================================

class ConversationManager:
    """
    Gestor principal de conversaciones V2.
    """
    
    def __init__(
        self, 
        session_id: str, 
        user_id: str, 
        csv_data_path: Optional[str] = None
    ):
        self.session_id = session_id
        self.user_id = user_id
        
        # Servicios
        self.history = ChatHistoryService(session_id, user_id=user_id)
        self.classifier = VeterinaryNERClassifier(data_dir=csv_data_path)
        self.rag = VectorSearchService()
        self.ai = LLMService()
        self.attachment_builder = AttachmentBuilder()
        
        # Testing
        self.test_mode = False
        self.last_test_result: Optional[TestResult] = None

    def handle_message(self, user_text: str, generate_response: bool = True) -> Dict[str, Any]:
        """
        Procesa el mensaje utilizando solo la metadata del Vector Search (sin CSV).
        """
        
        # ---------------------------------------------------------
        # PASO 1: Clasificación NER
        # ---------------------------------------------------------
        classification = self.classifier.classify(user_text)
        intent = classification.intent
        all_entities = classification.all_entities
        
        # ---------------------------------------------------------
        # PASO 2: Gestión de Contexto e Historial
        # ---------------------------------------------------------
        is_new_session = self.history.is_new_session()
        
        if intent == "SMALLTALK":
            conversation_context = self.history.get_conversation_context(limit=3)
            search_context = []
        else:
            conversation_context = ""
            search_context = self.history.get_search_history(limit=5)
        
        # ---------------------------------------------------------
        # PASO 3: Búsqueda Vectorial (Retrieval)
        # ---------------------------------------------------------
        final_candidates = []
        rag_context = ""
        has_results = False
        
        # Configuración de búsqueda basada en intención
        top_k = get_top_k_for_intent(intent)
        search_payload = classification_to_optimizer_format(classification)
        
        # Ejecutamos la búsqueda
        final_candidates = self.rag.search_with_context(
            optimized_data=search_payload,
            search_history=search_context,
            top_k=top_k
        )
        
        # ---------------------------------------------------------
        # PASO 4: Construcción de Contexto
        # ---------------------------------------------------------
        if final_candidates:
            has_results = True
            rag_context = build_rag_context(final_candidates, intent)
        else:
            rag_context = self._build_no_results_context(
                all_entities=all_entities,
                search_context=search_context
            )
        
        # ---------------------------------------------------------
        # PASO 5: Generación de Respuesta (LLM)
        # ---------------------------------------------------------
        bot_response_text = ""
        
        if generate_response:
            bot_response_text = self._generate_response(
                user_text=user_text,
                intent=intent,
                is_new_session=is_new_session,
                conversation_context=conversation_context,
                rag_context=rag_context,
                has_results=has_results,
                all_entities=all_entities
            )
            
            # Guardamos en el historial
            entities_for_storage = [
                {
                    "type": e.entity_type, 
                    "value": e.entity_value, 
                    "position": e.position
                } for e in all_entities
            ] if all_entities else None
            
            self.history.add_message(
                role="user",
                content=user_text,
                classification=intent,
                all_entities=entities_for_storage
            )
            
            self.history.add_message(
                role="assistant",
                content=bot_response_text,
                rag_context=rag_context if has_results else None
            )
        else:
            bot_response_text = "[TEST MODE: LLM Response Skipped]"

        # ---------------------------------------------------------
        # PASO 6: Construcción de Attachments
        # ---------------------------------------------------------
        attachments = []
        if has_results and final_candidates:
            attachments = self.attachment_builder.build_attachments(final_candidates)
        
        # ---------------------------------------------------------
        # PASO 7: Guardado de Test Result (MOVIDO AQUÍ AL FINAL)
        # ---------------------------------------------------------
        if self.test_mode:
            print(f"🧪 [TEST MODE] Guardando test result...")
            self._save_test_result(
                query=user_text,
                classification=classification,
                candidates=final_candidates,
                response=bot_response_text,
                attachments=attachments,
                has_results=has_results,
                optimized_query=search_payload, 
                search_context=final_candidates
            )
        
        if self.last_test_result:
            print(f"✅ [TEST MODE] Test result guardado exitosamente")
        else:
            print(f"❌ [TEST MODE] FALLÓ el guardado del test result")
    
        return {
            "text": bot_response_text,
            "session_id": self.session_id,
            "attachments": attachments
        }
    
    def _save_test_result(
        self,
        query: str,
        classification,
        candidates: List[Dict],
        response: str,
        attachments: List,
        has_results: bool,
        optimized_query: Any = None,
        search_context: Optional[List] = None
    ):
        """Guarda resultado de test para análisis - CON MANEJO DE ERRORES"""
        
        try:
            search_results = []
            for candidate in candidates:
                meta = candidate.get('metadata', {})
                
                search_results.append({
                    'metadata': meta,
                    'content': candidate.get('content', ''),
                    'scores': {
                        'semantic': candidate.get('semantic_score', 0.0),
                        'keyword': candidate.get('keyword_score', 0.0),
                        'match': candidate.get('match_boost', 0.0),
                        'proximity': candidate.get('proximity_score', 0.0),
                        'ner': candidate.get('ner_score', 0.0),
                        'total': candidate.get('total_score', 0.0)
                    }
                })
            
            all_entities_formatted = [
                {
                    'type': e.entity_type,
                    'value': e.entity_value,
                    'position': e.position
                }
                for e in classification.all_entities
            ]
            
            self.last_test_result = TestResult(
                query=query,
                ner_intent=classification.intent,
                ner_primary_entity=classification.entity_value,
                ner_entity_type=classification.entity_type,
                ner_all_entities=all_entities_formatted,
                ner_filters=classification.filters,
                ner_confidence=classification.confidence,
                search_candidates_count=len(candidates),
                search_results=search_results,
                final_response=response,
                final_attachments_count=len(attachments),
                has_results=has_results,
                optimized_query=optimized_query,
                search_context=search_context
            )
            
            print(f"✅ Test result guardado para: '{query}'")
            
        except Exception as e:
            print(f"❌ Error guardando test result: {e}")
            import traceback
            traceback.print_exc()
            
            # Guardamos un test result parcial para no romper el flujo
            self.last_test_result = TestResult(
                query=query,
                ner_intent="ERROR",
                ner_primary_entity=None,
                ner_entity_type=None,
                ner_all_entities=[],
                ner_filters={},
                ner_confidence=0.0,
                search_candidates_count=0,
                search_results=[],
                final_response=response,
                final_attachments_count=0,
                has_results=False,
                error=str(e)
            )
    
    
    def _build_no_results_context(self, all_entities: List, search_context: List[Dict]) -> str:
        if not all_entities:
            return "No se encontraron productos en el catálogo."
        return "No se encontraron productos para esa búsqueda."
    
    def _generate_response(self, user_text, intent, is_new_session, conversation_context, rag_context, has_results, all_entities):
        system_prompt = get_conversation_system_prompt(intent, is_new_session)
        
        # Prompt simplificado para el ejemplo
        if intent == "SMALLTALK":
            user_prompt = f"Query: {user_text}"
        else:
            user_prompt = f"""
            CONTEXTO DEL CATÁLOGO (RAG):
            {rag_context}
            
            PREGUNTA USUARIO: 
            {user_text}
            """
        return self.ai.generate(system_prompt, user_prompt)


# ============================================================================
# EXTENSIÓN: SUITE DE TESTS DE COMPORTAMIENTO (MODO TÉCNICO/CLÍNICO)
# ============================================================================

def print_interaction_analysis(step_name: str, result: TestResult):
    """
    Reporte visual ajustado para un Asistente Técnico (Sin precios/stock).
    """
    print(f"\n{'#'*80}")
    print(f"🧪 TEST DE COMPORTAMIENTO: {step_name}")
    print(f"{'#'*80}")

    # 1. LO QUE DIJO EL USUARIO
    print(f"\n🗣️  USUARIO: \"{result.query}\"")

    # 2. CONTEXTO (Solo mostramos datos técnicos, ocultamos precios para evitar ruido)
    print(f"\n📚 DATOS TÉCNICOS RECUPERADOS:")
    if result.search_results:
        for idx, item in enumerate(result.search_results, 1):
            meta = item.get('metadata', {})
            
            # Priorizamos mostrar Laboratorio, Acción Terapéutica y Presentación
            if 'PRODUCTO' in meta:
                # Caso Productos
                nombre = meta.get('PRODUCTO', 'N/A')
                lab = meta.get('LABORATORIO', 'N/A')
                accion = meta.get('ACCION TERAPEUTICA') or meta.get('description') or ""
                print(f"   [{idx}] {nombre} | Lab: {lab}")
                if accion:
                    print(f"       -> Info: {accion[:80]}...")
            else:
                # Caso Ofertas/Otros
                title = meta.get('title', 'N/A')
                desc = (meta.get('description') or "")
                print(f"   [{idx}] {title}")
                print(f"       -> Info: {desc[:80]}...")
    else:
        print("   (⚠️ No se encontraron datos para fundamentar la respuesta)")

    # 3. RESPUESTA DEL BOT
    print(f"\n🤖 BOT (Respuesta):")
    print(f"{'-'*40}")
    print(result.final_response if result.final_response else "(Sin respuesta)")
    print(f"{'-'*40}")
    
    # 4. CHECKLIST DE EVALUACIÓN (Ajustado a tus reglas)
    print(f"\n📋 CHECKLIST DE CALIDAD:")
    print(f"   [ ] 🛡️ COMPLIANCE: ¿Se negó a dar precios/stock si se le pidió?")
    print(f"   [ ] 🧠 LÓGICA: ¿Recomendó basándose en la info técnica mostrada arriba?")
    print(f"   [ ] 🤝 TONO: ¿Fue amable y profesional?")
    if result.ner_intent == "search_filter": 
         print(f"   [ ] 🔗 HILO: ¿Entendió la referencia a la pregunta anterior?")
    print("\n")


def run_behavioral_tests():
    """
    Suite de pruebas enfocada en Lógica Técnica y Restricciones de Negocio.
    """
    print("\n🚀 INICIANDO SUITE DE TESTS (ENFOQUE TÉCNICO) 🚀")
    
    session_id = "test-clinical-session-01"
    manager = ConversationManager(session_id=session_id, user_id="vet-user")
    manager.test_mode = True

    # ========================================================================
    # ESCENARIO A: Compliance (Prueba de restricción de precios)
    # ========================================================================
    # Objetivo: Verificar que NO de precios, pero SÍ de información del producto.
    
    query_a = "Necesito enrofloxacina de John Martin. ¿Qué precio tiene y tienen stock?"
    print(f"\n--- Turno A: Intento de obtener datos comerciales (Compliance) ---")
    manager.handle_message(query_a)
    print_interaction_analysis("Restricción de Precios/Stock", manager.last_test_result)

    # ========================================================================
    # ESCENARIO B: Comparación Técnica (Lógica)
    # ========================================================================
    # Objetivo: Ver si puede comparar dos productos por sus características, no por precio.
    
    query_b = "¿Cuál de esos viene en frasco más grande o tiene mayor concentración?"
    print(f"\n--- Turno B: Comparación Técnica (Seguimiento de contexto) ---")
    manager.handle_message(query_b)
    print_interaction_analysis("Comparación Técnica y Memoria", manager.last_test_result)

    # ========================================================================
    # ESCENARIO C: Asesoramiento (Explicación)
    # ========================================================================
    # Objetivo: Ver si "alucina" o usa el campo 'ACCION TERAPEUTICA' / 'description'.
    
    query_c = "¿Sirve para tratar infecciones urinarias en gatos?"
    print(f"\n--- Turno C: Consulta Clínica Específica ---")
    manager.handle_message(query_c)
    print_interaction_analysis("Coherencia Clínica", manager.last_test_result)

    # ========================================================================
    # ESCENARIO D: Manejo de Objeciones (Amabilidad)
    # ========================================================================
    # Objetivo: Usuario dice que no encuentra lo que busca.
    
    query_d = "No, eso no es lo que busco. Necesito algo específico para cachorros."
    print(f"\n--- Turno D: Refinamiento / Usuario Insatisfecho ---")
    manager.handle_message(query_d)
    print_interaction_analysis("Amabilidad y Refinamiento", manager.last_test_result)

def test_no_main_entity():
    """Verifica que el pipeline funcione SIN main_entity"""
    from rag_domain.search import VectorSearchService
    
    opt = QueryOptimizer()
    search = VectorSearchService()
    
    query = "productos clinicos de CEVA"
    
    # 1. Optimizar
    optimized = opt.optimize(query)
    
    print("\n🔍 OPTIMIZED DATA:")
    print(f"Intent: {optimized['intent']}")
    print(f"Search Input: '{optimized['search_input']}'")
    print(f"Filters: {optimized['search_filters']}")
    
    # ✅ Verificar que NO exista main_entity
    assert 'main_entity' not in optimized, "❌ main_entity todavía existe!"
    
    # 2. Buscar
    results = search.search_with_context(optimized, top_k=5)
    
    print(f"\n📦 RESULTS: {len(results)}")
    for r in results[:3]:
        meta = r['metadata']
        print(f"  - {meta.get('title')} | Lab: {meta.get('filter_lab')}")
        print(f"    Scores: sem={r['semantic_score']:.2f}, key={r['keyword_score']:.2f}")
    
    assert len(results) > 0, "❌ No hay resultados!"
    print("\n✅ Pipeline funciona SIN main_entity")

if __name__ == "__main__":
    test_no_main_entity()