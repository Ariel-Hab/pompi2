"""
CONVERSATION MANAGER V5 - Query Architect Integration
Integración completa del sistema:
- NER High Recall
- LLM Query Architect
- Search con exclusiones
- Memoria mejorada
"""
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from chat_domain.chat_history import ChatHistoryService
from rag_domain.ner_classifier import VeterinaryNERClassifier
from rag_domain.optimizer import QueryOptimizer
from rag_domain.prompts import build_rag_context, get_conversation_system_prompt
from ai_gateway.llm_client import LLMService
from chat_domain.attachment_builder import AttachmentBuilder
from rag_domain.config import get_top_k_for_intent
from rag_domain.search import VectorSearchService


@dataclass
class TestResult:
    """Resultado de test con arquitectura Query Architect"""
    query: str
    
    # NER Detection
    ner_intent: str
    ner_candidates_count: int
    ner_all_candidates: List[Dict]
    ner_filters: Dict
    ner_confidence: float
    
    # Query Architect Results
    architect_search_term: str
    architect_filters: Dict
    architect_exclusions: List[str]
    architect_relevant_history: List[Dict]
    
    # Search Results
    search_candidates_count: int
    search_results: List[Dict]
    
    # Final Output
    final_response: str
    final_attachments_count: int
    
    # Metadata
    has_results: bool
    optimized_query: Any = None
    error: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)
    
    def print_summary(self):
        """Resumen visual detallado"""
        print(f"\n{'='*80}")
        print(f"🧪 ANÁLISIS: '{self.query}'")
        print(f"{'='*80}")
        
        # 1. NER
        print(f"\n🎯 1. NER (HIGH RECALL):")
        print(f"   Intent: {self.ner_intent}")
        print(f"   Candidatos: {self.ner_candidates_count}")
        if self.ner_all_candidates:
            print(f"   Top Candidates:")
            for idx, c in enumerate(self.ner_all_candidates[:5], 1):
                print(f"     {idx}. {c['type']}: {c['value']} (score: {c.get('score', 0):.2f})")
        
        # 2. Query Architect
        print(f"\n⚖️ 2. QUERY ARCHITECT:")
        print(f"   Search Term (Vectorial): '{self.architect_search_term}'")
        print(f"   Filtros Estructurados:")
        for key, val in self.architect_filters.items():
            if key != 'exclude_brands':
                print(f"     • {key}: {val}")
        if self.architect_exclusions:
            print(f"   ❌ Exclusiones: {self.architect_exclusions}")
        if self.architect_relevant_history:
            print(f"   📜 Historial Adjunto: {len(self.architect_relevant_history)} objetos")
        
        # 3. Search Results
        print(f"\n📦 3. RESULTADOS DE BÚSQUEDA ({self.search_candidates_count}):")
        if self.search_results:
            for idx, result in enumerate(self.search_results[:5], 1):
                meta = result.get('metadata', {})
                nombre = meta.get('PRODUCTO') or meta.get('title', 'N/A')
                lab = meta.get('LABORATORIO') or meta.get('filter_lab', 'N/A')
                score = result.get('total_score', 0)
                print(f"   #{idx} {nombre}")
                print(f"       Lab: {lab} | Score: {score:.4f}")
                
                # Mostrar si fue excluido (no debería aparecer)
                if lab in self.architect_exclusions:
                    print(f"       ⚠️ WARNING: Este producto debería estar excluido!")
        else:
            print("   (Sin resultados)")
        
        # 4. Response
        print(f"\n💬 4. RESPUESTA FINAL:")
        print(f"{'─'*80}")
        print(self.final_response[:500] + "..." if len(self.final_response) > 500 else self.final_response)
        print(f"{'─'*80}")
        print(f"📎 Attachments: {self.final_attachments_count}")


class ConversationManager:
    """
    Gestor V5: Query Architect Integration
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
        self.ner = VeterinaryNERClassifier(data_dir=csv_data_path)
        self.optimizer = QueryOptimizer()
        self.rag = VectorSearchService()
        self.ai = LLMService()
        self.attachment_builder = AttachmentBuilder()
        
        # Testing
        self.test_mode = False
        self.last_test_result: Optional[TestResult] = None

    def handle_message(self, user_text: str, generate_response: bool = True) -> Dict[str, Any]:
        """
        Pipeline V5: NER → Query Architect → Search → Response
        """
        
        print(f"\n{'='*80}")
        print(f"🚀 PROCESANDO: '{user_text}'")
        print(f"{'='*80}")
        
        # ═══════════════════════════════════════════════════════════
        # PASO 1: NER High Recall
        # ═══════════════════════════════════════════════════════════
        classification = self.ner.classify(user_text)
        intent = classification.intent
        all_candidates = classification.all_entities
        
        print(f"\n🎯 [PASO 1] NER detectó {len(all_candidates)} candidatos")
        print(f"   Intent: {intent}")
        
        # ═══════════════════════════════════════════════════════════
        # PASO 2: Obtener Historial
        # ═══════════════════════════════════════════════════════════
        is_new_session = self.history.is_new_session()
        
        if intent == "SMALLTALK":
            conversation_context = self.history.get_conversation_context(limit=3)
            search_history = []
        else:
            conversation_context = ""
            search_history = self.history.get_search_history(limit=5)
        
        print(f"\n📜 [PASO 2] Historial: {len(search_history)} búsquedas previas")
        
        # ═══════════════════════════════════════════════════════════
        # PASO 3: Query Architect (LLM)
        # ═══════════════════════════════════════════════════════════
        optimized_data = self.optimizer.optimize(
            user_text, 
            classification,
            search_history=search_history
        )
        
        search_term = optimized_data.get('search_input', '')
        filters = optimized_data.get('search_filters', {})
        exclusions = filters.get('exclude_brands', [])
        relevant_history = optimized_data.get('relevant_history', [])
        
        print(f"\n⚖️ [PASO 3] Query Architect:")
        print(f"   Search Term: '{search_term}'")
        print(f"   Filtros: {list(filters.keys())}")
        if exclusions:
            print(f"   ❌ Exclusiones: {exclusions}")
        if relevant_history:
            print(f"   📜 Historial adjunto: {len(relevant_history)} objetos")
        
        # ═══════════════════════════════════════════════════════════
        # PASO 4: Vector Search con Exclusiones
        # ═══════════════════════════════════════════════════════════
        final_candidates = []
        rag_context = ""
        has_results = False
        
        top_k = get_top_k_for_intent(intent)
        
        if intent != "SMALLTALK":
            print(f"\n🔍 [PASO 4] Ejecutando búsqueda (top_k={top_k})...")
            
            final_candidates = self.rag.search_with_context(
                optimized_data=optimized_data,
                search_history=search_history,
                top_k=top_k
            )
            
            print(f"   Resultados: {len(final_candidates)} productos")
            
            if final_candidates:
                has_results = True
                rag_context = build_rag_context(final_candidates, intent)
                
                # Verificar exclusiones
                if exclusions:
                    excluded_found = []
                    for candidate in final_candidates:
                        meta = candidate.get('metadata', {})
                        lab = (meta.get('LABORATORIO') or meta.get('filter_lab', '')).lower()
                        if any(excl.lower() in lab for excl in exclusions):
                            excluded_found.append(meta.get('PRODUCTO', 'N/A'))
                    
                    if excluded_found:
                        print(f"   ⚠️ WARNING: Productos excluidos encontrados: {excluded_found}")
            else:
                rag_context = self._build_no_results_context(
                    search_term, 
                    filters, 
                    exclusions
                )
        
        # ═══════════════════════════════════════════════════════════
        # PASO 5: Contexto de Historial
        # ═══════════════════════════════════════════════════════════
        history_objects_context = ""
        if relevant_history:
            history_objects_context = self._build_history_context(
                relevant_history, 
                search_history
            )
        
        # ═══════════════════════════════════════════════════════════
        # PASO 6: Generación de Respuesta
        # ═══════════════════════════════════════════════════════════
        bot_response_text = ""
        
        if generate_response:
            print(f"\n💬 [PASO 6] Generando respuesta...")
            
            bot_response_text = self._generate_response(
                user_text=user_text,
                intent=intent,
                is_new_session=is_new_session,
                conversation_context=conversation_context,
                rag_context=rag_context,
                history_objects_context=history_objects_context,
                has_results=has_results,
                search_term=search_term,
                filters=filters,
                exclusions=exclusions
            )
            
            # Guardar en historial
            entities_for_storage = [
                {
                    "type": c.entity_type,
                    "value": c.entity_value,
                    "position": c.position,
                    "score": c.score
                } for c in all_candidates[:10]
            ]
            
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
        
        # ═══════════════════════════════════════════════════════════
        # PASO 7: Attachments
        # ═══════════════════════════════════════════════════════════
        attachments = []
        if has_results and final_candidates:
            attachments = self.attachment_builder.build_attachments(final_candidates)
        
        # ═══════════════════════════════════════════════════════════
        # PASO 8: Test Result
        # ═══════════════════════════════════════════════════════════
        if self.test_mode:
            self._save_test_result(
                query=user_text,
                classification=classification,
                optimized_data=optimized_data,
                candidates=final_candidates,
                response=bot_response_text,
                attachments=attachments,
                has_results=has_results
            )
        
        print(f"\n✅ Procesamiento completado")
        print(f"{'='*80}\n")
        
        return {
            "text": bot_response_text,
            "session_id": self.session_id,
            "attachments": attachments,
            "debug": {
                "ner_candidates": len(all_candidates),
                "search_term": search_term,
                "filters_applied": list(filters.keys()),
                "exclusions": exclusions,
                "search_results": len(final_candidates),
                "history_attached": len(relevant_history)
            }
        }
    
    def _build_history_context(
        self, 
        relevant_history: List[Dict], 
        full_search_history: List[Dict]
    ) -> str:
        """
        Construye contexto de objetos del historial.
        """
        if not relevant_history:
            return ""
        
        context_lines = ["\n--- CONTEXTO DE BÚSQUEDAS PREVIAS ---"]
        
        for item in relevant_history:
            entity_name = item.get('entity', 'N/A')
            reason = item.get('reason', 'Referencia del historial')
            
            # Buscar detalles
            details = next(
                (h for h in full_search_history if h.get('entity') == entity_name),
                None
            )
            
            if details:
                all_entities = details.get('all_entities', [])
                entities_str = ", ".join([
                    e.get('value', '') for e in all_entities[:3]
                ])
                context_lines.append(
                    f"\nBúsqueda anterior: {entity_name}"
                    f"\n  Razón: {reason}"
                    f"\n  También: {entities_str}"
                )
        
        context_lines.append("--- FIN CONTEXTO PREVIO ---\n")
        return "\n".join(context_lines)
    
    def _save_test_result(
        self,
        query: str,
        classification,
        optimized_data: Dict,
        candidates: List[Dict],
        response: str,
        attachments: List,
        has_results: bool
    ):
        """Guarda resultado de test"""
        try:
            # Formatear candidatos del NER
            ner_candidates_formatted = [
                {
                    'type': c.entity_type,
                    'value': c.entity_value,
                    'score': c.score
                }
                for c in classification.all_entities[:20]
            ]
            
            # Extraer datos del Architect
            filters = optimized_data.get('search_filters', {})
            
            # Formatear resultados
            search_results = []
            for candidate in candidates:
                search_results.append({
                    'metadata': candidate.get('metadata', {}),
                    'content': candidate.get('content', ''),
                    'total_score': candidate.get('total_score', 0.0),
                    'scores': {
                        'semantic': candidate.get('semantic_score', 0.0),
                        'keyword': candidate.get('keyword_score', 0.0),
                        'ner': candidate.get('ner_similarity_score', 0.0),
                        'family': candidate.get('family_score', 0.0),
                        'exact': candidate.get('exact_match_score', 0.0)
                    }
                })
            
            self.last_test_result = TestResult(
                query=query,
                ner_intent=classification.intent,
                ner_candidates_count=len(classification.all_entities),
                ner_all_candidates=ner_candidates_formatted,
                ner_filters=classification.filters,
                ner_confidence=classification.confidence,
                architect_search_term=optimized_data.get('search_input', ''),
                architect_filters=filters,
                architect_exclusions=filters.get('exclude_brands', []),
                architect_relevant_history=optimized_data.get('relevant_history', []),
                search_candidates_count=len(candidates),
                search_results=search_results,
                final_response=response,
                final_attachments_count=len(attachments),
                has_results=has_results,
                optimized_query=optimized_data
            )
            
            print(f"✅ Test result guardado")
            
        except Exception as e:
            print(f"❌ Error guardando test result: {e}")
            import traceback
            traceback.print_exc()
    
    def _build_no_results_context(
        self, 
        search_term: str, 
        filters: Dict, 
        exclusions: List[str]
    ) -> str:
        """Contexto cuando no hay resultados"""
        parts = []
        
        if search_term:
            parts.append(f"búsqueda: '{search_term}'")
        
        if filters.get('brand'):
            parts.append(f"marca: {filters['brand']}")
        
        if filters.get('category'):
            parts.append(f"categoría: {filters['category']}")
        
        if exclusions:
            parts.append(f"excluyendo: {', '.join(exclusions)}")
        
        if parts:
            return f"No se encontraron productos para {', '.join(parts)}."
        
        return "No se encontraron productos en el catálogo."
    
    def _generate_response(
        self, 
        user_text: str, 
        intent: str, 
        is_new_session: bool,
        conversation_context: str, 
        rag_context: str, 
        history_objects_context: str,
        has_results: bool, 
        search_term: str,
        filters: Dict,
        exclusions: List[str]
    ):
        """Genera respuesta del LLM"""
        system_prompt = get_conversation_system_prompt(intent, is_new_session)
        
        if intent == "SMALLTALK":
            user_prompt = f"Query: {user_text}"
        else:
            # Combinar contextos
            full_context = []
            
            if history_objects_context:
                full_context.append(history_objects_context)
            
            if rag_context:
                full_context.append(rag_context)
            
            context_text = "\n\n".join(full_context) if full_context else "Sin resultados"
            
            # Información de la consulta estructurada
            query_info_parts = [f"Búsqueda: '{search_term}'"]
            
            if filters.get('brand'):
                query_info_parts.append(f"Marca: {filters['brand']}")
            if filters.get('category'):
                query_info_parts.append(f"Categoría: {filters['category']}")
            if filters.get('species'):
                query_info_parts.append(f"Especie: {filters['species']}")
            if filters.get('presentation'):
                query_info_parts.append(f"Presentación: {filters['presentation']}")
            if exclusions:
                query_info_parts.append(f"❌ Excluyendo: {', '.join(exclusions)}")
            
            query_info = " | ".join(query_info_parts)
            
            user_prompt = f"""
{context_text}

CONSULTA ESTRUCTURADA:
{query_info}

PREGUNTA ORIGINAL DEL USUARIO: 
{user_text}

IMPORTANTE: 
- Responde SOLO con productos del catálogo mostrado arriba.
- Si hay exclusiones (❌), NO menciones esas marcas bajo ninguna circunstancia.
- Sé selectivo: si el usuario pidió algo específico (ej: "10kg"), NO ofrezcas otros pesos.
"""
        
        return self.ai.generate(system_prompt, user_prompt)


# ═══════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════

def test_query_architect_integration():
    """Test completo del sistema integrado"""
    print("\n🚀 TEST: QUERY ARCHITECT INTEGRATION")
    print("="*80)
    
    manager = ConversationManager(
        session_id="test-architect-v5",
        user_id="vet-test"
    )
    manager.test_mode = True
    
    # Test 1: Consulta con exclusión
    print("\n" + "="*80)
    print("📝 TEST 1: Consulta con exclusión")
    print("="*80)
    query1 = "antiparasitarios para perros pero que no sea bravecto"
    manager.handle_message(query1, generate_response=False)
    
    if manager.last_test_result:
        manager.last_test_result.print_summary()
        
        # Validaciones
        assert 'Bravecto' not in str(manager.last_test_result.search_results), \
            "❌ FALLO: Bravecto NO debería aparecer en los resultados"
        print("\n✅ PASS: Exclusión funcionó correctamente")
    
    # Test 2: Consulta específica con peso
    print("\n\n" + "="*80)
    print("📝 TEST 2: Consulta específica con peso")
    print("="*80)
    query2 = "power gold de 10kg"
    manager.handle_message(query2, generate_response=False)
    
    if manager.last_test_result:
        manager.last_test_result.print_summary()
        
        # Validaciones
        assert manager.last_test_result.architect_search_term == "Power Gold", \
            f"❌ FALLO: Search term debería ser 'Power Gold', fue '{manager.last_test_result.architect_search_term}'"
        assert manager.last_test_result.architect_filters.get('weight_min') == 10, \
            "❌ FALLO: No se detectó el peso correctamente"
        print("\n✅ PASS: Consulta específica correcta")
    
    # Test 3: Refinamiento con historial
    print("\n\n" + "="*80)
    print("📝 TEST 3: Refinamiento con historial")
    print("="*80)
    query3 = "y en gotas?"
    manager.handle_message(query3, generate_response=False)
    
    if manager.last_test_result:
        manager.last_test_result.print_summary()
        
        # Validaciones
        assert 'gotas' in str(manager.last_test_result.architect_filters).lower(), \
            "❌ FALLO: No se aplicó el filtro de presentación"
        print("\n✅ PASS: Refinamiento con historial funciona")
    
    print("\n" + "="*80)
    print("🎉 TODOS LOS TESTS PASARON")
    print("="*80)


if __name__ == "__main__":
    test_query_architect_integration()