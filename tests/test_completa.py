"""
TEST SUITE COMPLETA - Query Architect System
Visualización detallada de cada módulo y su procesamiento
"""
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Imports (ajustar según tu estructura)
from rag_domain.ner_classifier import VeterinaryNERClassifier, ClassificationResult
from rag_domain.optimizer import QueryOptimizer
from rag_domain.search import VectorSearchService
from chat_domain.chat_history import ChatHistoryService


# ═══════════════════════════════════════════════════════════════
# VISUALIZADOR DE PIPELINE
# ═══════════════════════════════════════════════════════════════

class PipelineVisualizer:
    """
    Visualiza cada paso del pipeline con máximo detalle.
    """
    
    def __init__(self):
        self.ner = VeterinaryNERClassifier()
        self.optimizer = QueryOptimizer()
        self.search = VectorSearchService()
        self.session_id = f"test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.history = ChatHistoryService(self.session_id, user_id="test-user")
    
    def test_complete_pipeline(self, query: str, search_history: List[Dict] = None):
        """
        Ejecuta y visualiza el pipeline completo con MÁXIMO detalle.
        """
        print("\n" + "═"*100)
        print(f"🧪 TESTING QUERY: '{query}'")
        print("═"*100)
        
        # ════════════════════════════════════════════════════════════════
        # PASO 1: NER CLASSIFIER
        # ════════════════════════════════════════════════════════════════
        print("\n" + "─"*100)
        print("📍 PASO 1: NER CLASSIFIER (High Recall)")
        print("─"*100)
        
        classification = self.ner.classify(query)
        
        self._visualize_ner_output(classification)
        
        # ════════════════════════════════════════════════════════════════
        # PASO 2: QUERY OPTIMIZER (LLM Judge)
        # ════════════════════════════════════════════════════════════════
        print("\n" + "─"*100)
        print("📍 PASO 2: QUERY OPTIMIZER (LLM Query Architect)")
        print("─"*100)
        
        optimized_data = self.optimizer.optimize(
            query,
            classification,
            search_history=search_history or []
        )
        
        self._visualize_optimizer_output(optimized_data)
        
        # ════════════════════════════════════════════════════════════════
        # PASO 3: VECTOR SEARCH
        # ════════════════════════════════════════════════════════════════
        print("\n" + "─"*100)
        print("📍 PASO 3: VECTOR SEARCH (Hybrid Search)")
        print("─"*100)
        
        results = self.search.search_with_context(
            optimized_data=optimized_data,
            search_history=search_history or [],
            top_k=5
        )
        
        self._visualize_search_output(results)
        
        # ════════════════════════════════════════════════════════════════
        # PASO 4: DATOS FINALES PARA CHAT MANAGER
        # ════════════════════════════════════════════════════════════════
        print("\n" + "─"*100)
        print("📍 PASO 4: DATOS PARA CHAT MANAGER (Input del Prompt Final)")
        print("─"*100)
        
        self._visualize_final_data_for_manager(
            query,
            optimized_data,
            results,
            classification.intent
        )
        
        print("\n" + "═"*100)
        print("✅ PIPELINE COMPLETO")
        print("═"*100)
        
        return {
            'ner_output': classification,
            'optimizer_output': optimized_data,
            'search_output': results
        }
    
    def _visualize_ner_output(self, classification: ClassificationResult):
        """Visualiza output del NER con máximo detalle."""
        
        print("\n🎯 OUTPUT DEL NER:")
        print(f"   Intent: {classification.intent}")
        print(f"   Confidence: {classification.confidence:.2f}")
        
        print("\n📋 ENTIDADES DETECTADAS ({} total):".format(len(classification.all_entities)))
        
        # Agrupar por tipo
        by_type = {}
        for entity in classification.all_entities:
            if entity.entity_type not in by_type:
                by_type[entity.entity_type] = []
            by_type[entity.entity_type].append(entity)
        
        for entity_type, entities in sorted(by_type.items()):
            print(f"\n   🏷️  {entity_type}:")
            for idx, e in enumerate(entities[:5], 1):  # Top 5 por tipo
                print(f"      {idx}. '{e.entity_value}' "
                      f"(score: {e.score:.2f}, pos: {e.position})")
            if len(entities) > 5:
                print(f"      ... y {len(entities) - 5} más")
        
        print("\n🔧 FILTROS EXTRAÍDOS:")
        if classification.filters:
            for key, value in classification.filters.items():
                print(f"   • {key}: {value}")
        else:
            print("   (Sin filtros)")
        
        print("\n📊 RESUMEN NER:")
        print(f"   ├─ Total entidades: {len(classification.all_entities)}")
        print(f"   ├─ Tipos detectados: {len(by_type)}")
        print(f"   ├─ Intent: {classification.intent}")
        print(f"   └─ Filtros: {len(classification.filters)} extraídos")
    
    def _visualize_optimizer_output(self, optimized_data: Dict):
        """Visualiza output del Optimizer con máximo detalle."""
        
        print("\n⚖️ OUTPUT DEL OPTIMIZER:")
        print(f"   Intent: {optimized_data.get('intent')}")
        
        print("\n🔍 SEARCH TERM (Para búsqueda vectorial):")
        search_term = optimized_data.get('search_input', '')
        print(f"   📝 '{search_term}'")
        print(f"   📏 Longitud: {len(search_term)} caracteres")
        
        print("\n🎛️ FILTROS ESTRUCTURADOS:")
        filters = optimized_data.get('search_filters', {})
        
        if not filters:
            print("   (Sin filtros)")
        else:
            # Categorizar filtros
            identity_filters = ['brand', 'laboratorios', 'target_products', 'categorias']
            attribute_filters = ['weight_min', 'weight_max', 'weight_unit', 'presentation', 'species', 'category', 'drug']
            commercial_filters = ['is_offer', 'is_transfer']
            negative_filters = ['exclude_brands']
            
            print("\n   📌 IDENTIDAD:")
            for key in identity_filters:
                if key in filters:
                    value = filters[key]
                    if isinstance(value, list):
                        print(f"      • {key}: {', '.join(value)}")
                    else:
                        print(f"      • {key}: {value}")
            
            print("\n   🎚️  ATRIBUTOS:")
            for key in attribute_filters:
                if key in filters:
                    print(f"      • {key}: {filters[key]}")
            
            print("\n   🏷️  COMERCIALES:")
            for key in commercial_filters:
                if key in filters:
                    print(f"      • {key}: {filters[key]}")
            
            print("\n   ❌ EXCLUSIONES:")
            for key in negative_filters:
                if key in filters:
                    exclusions = filters[key]
                    if exclusions:
                        print(f"      • {key}: {', '.join(exclusions)}")
        
        print("\n🧠 DEBUG ANALYSIS (Del LLM):")
        debug = optimized_data.get('debug_analysis', {})
        
        if debug:
            print(f"   ├─ Approved Entities: {debug.get('approved_entities', [])}")
            print(f"   ├─ Excluded Brands: {debug.get('excluded_brands', [])}")
            print(f"   └─ Relevant History: {len(debug.get('relevant_history_objects', []))} objetos")
        
        print("\n📊 RESUMEN OPTIMIZER:")
        print(f"   ├─ Search term definido: {'✅' if search_term else '❌'}")
        print(f"   ├─ Filtros aplicados: {len(filters)}")
        print(f"   ├─ Tiene exclusiones: {'✅' if filters.get('exclude_brands') else '❌'}")
        print(f"   └─ Usa historial: {'✅' if optimized_data.get('relevant_history') else '❌'}")
    
    def _visualize_search_output(self, results: List[Dict]):
        """Visualiza output del Search con scoring detallado."""
        
        print("\n🔎 OUTPUT DE LA BÚSQUEDA:")
        print(f"   Resultados obtenidos: {len(results)}")
        
        if not results:
            print("   ❌ Sin resultados")
            return
        
        print("\n📦 PRODUCTOS ENCONTRADOS:")
        
        for idx, result in enumerate(results, 1):
            meta = result.get('metadata', {})
            debug = result.get('_debug', {})
            
            # Header
            print(f"\n   ╔═══ RESULTADO #{idx} ═══════════════════════════════════")
            
            # Identidad
            producto = meta.get('PRODUCTO') or meta.get('title', 'N/A')
            lab = meta.get('LABORATORIO') or meta.get('filter_lab', 'N/A')
            print(f"   ║ 📦 Producto: {producto}")
            print(f"   ║ 🏭 Laboratorio: {lab}")
            
            # Atributos
            categoria = meta.get('CATEGORIA', 'N/A')
            concepto = meta.get('CONCEPTO', 'N/A')
            especie = meta.get('ESPECIE', 'N/A')
            
            print(f"   ║ 🏷️  Categoría: {categoria}")
            print(f"   ║ 💊 Presentación: {concepto}")
            print(f"   ║ 🐾 Especie: {especie}")
            
            # Flags comerciales
            flags = []
            if self._is_true(meta.get('is_offer')):
                flags.append("🏷️ OFERTA")
            if self._is_true(meta.get('has_transfer')):
                flags.append("🎁 TRANSFER")
            
            if flags:
                print(f"   ║ 🎉 Promociones: {', '.join(flags)}")
            
            # SCORING DETALLADO
            total_score = result.get('total_score', 0)
            print(f"   ║")
            print(f"   ║ 📊 SCORE TOTAL: {total_score:.4f}")
            print(f"   ║    ├─ Semantic:     {debug.get('sem', 0):.4f}")
            print(f"   ║    ├─ Keyword FTS:  {debug.get('key', 0):.4f}")
            print(f"   ║    ├─ NER Similar:  {debug.get('ner', 0):.4f}")
            print(f"   ║    ├─ Family Match: {debug.get('fam', 0):.4f}")
            print(f"   ║    ├─ Exact Match:  {debug.get('exact', 0):.4f}")
            print(f"   ║    ├─ Penalty:      {debug.get('penalty', 0):.4f}")
            print(f"   ║    └─ Boost:        {debug.get('boost', 0):.4f}")
            
            if debug.get('attr_matches', 0) > 0:
                print(f"   ║       (Atributos coincidentes: {debug.get('attr_matches')})")
            
            print(f"   ╚═════════════════════════════════════════════════════")
        
        print("\n📊 RESUMEN BÚSQUEDA:")
        print(f"   ├─ Total resultados: {len(results)}")
        
        # Score range
        if results:
            scores = [r.get('total_score', 0) for r in results]
            print(f"   ├─ Score máximo: {max(scores):.4f}")
            print(f"   ├─ Score mínimo: {min(scores):.4f}")
            print(f"   └─ Diferencia: {max(scores) - min(scores):.4f}")
    
    def _visualize_final_data_for_manager(
        self,
        query: str,
        optimized_data: Dict,
        results: List[Dict],
        intent: str
    ):
        """
        Visualiza EXACTAMENTE lo que recibe el Chat Manager para construir el prompt.
        """
        
        print("\n🎯 DATOS QUE RECIBE EL CHAT MANAGER:")
        
        # 1. Query original
        print(f"\n📝 1. QUERY ORIGINAL:")
        print(f"   '{query}'")
        
        # 2. Intent
        print(f"\n🎭 2. INTENT DETECTADO:")
        print(f"   {intent}")
        
        # 3. Search filters
        print(f"\n🔧 3. FILTROS APLICADOS:")
        filters = optimized_data.get('search_filters', {})
        
        if filters.get('exclude_brands'):
            print(f"   ❌ EXCLUSIONES: {', '.join(filters['exclude_brands'])}")
        
        if filters.get('brand') or filters.get('laboratorios'):
            brand = filters.get('brand') or (filters.get('laboratorios', [])[0] if filters.get('laboratorios') else None)
            print(f"   🏭 Marca/Lab: {brand}")
        
        if filters.get('category') or filters.get('categorias'):
            cat = filters.get('category') or (filters.get('categorias', [])[0] if filters.get('categorias') else None)
            print(f"   🏷️  Categoría: {cat}")
        
        if filters.get('species'):
            print(f"   🐾 Especie: {filters['species']}")
        
        if filters.get('presentation'):
            print(f"   💊 Presentación: {filters['presentation']}")
        
        if filters.get('weight_min') or filters.get('weight_max'):
            weight_str = f"{filters.get('weight_min', 0)}-{filters.get('weight_max', '∞')}{filters.get('weight_unit', 'kg')}"
            print(f"   ⚖️  Peso: {weight_str}")
        
        if filters.get('is_offer'):
            print(f"   🏷️  SOLO OFERTAS")
        
        if filters.get('is_transfer'):
            print(f"   🎁 SOLO TRANSFERS")
        
        # 4. Resultados (para RAG context)
        print(f"\n📦 4. PRODUCTOS PARA EL CONTEXTO RAG:")
        print(f"   Total: {len(results)} productos")
        
        if results:
            print(f"\n   Top 3 (los que verá el LLM en el prompt):")
            for idx, result in enumerate(results[:3], 1):
                meta = result.get('metadata', {})
                producto = meta.get('PRODUCTO') or meta.get('title', 'N/A')
                lab = meta.get('LABORATORIO', 'N/A')
                
                flags = []
                if self._is_true(meta.get('is_offer')):
                    flags.append("OFERTA")
                if self._is_true(meta.get('has_transfer')):
                    flags.append("TRANSFER")
                
                flags_str = f" [{', '.join(flags)}]" if flags else ""
                print(f"      {idx}. {producto} | Lab: {lab}{flags_str}")
        
        # 5. Contexto RAG simulado
        print(f"\n📄 5. CONTEXTO RAG (Texto que ve el LLM):")
        print("   " + "─"*70)
        
        if results:
            rag_context = self._build_rag_context_preview(results[:3])
            for line in rag_context.split('\n')[:20]:  # Primeras 20 líneas
                print(f"   {line}")
            if len(rag_context.split('\n')) > 20:
                print("   ...")
        else:
            print("   (Sin resultados en el catálogo)")
        
        print("   " + "─"*70)
        
        # 6. Estructura final para manager
        print(f"\n🎁 6. ESTRUCTURA COMPLETA PARA MANAGER:")
        
        final_structure = {
            "user_query": query,
            "intent": intent,
            "search_term": optimized_data.get('search_input'),
            "filters_applied": {
                k: v for k, v in filters.items() 
                if k not in ['target_products', 'laboratorios', 'categorias']
            },
            "exclusions": filters.get('exclude_brands', []),
            "results_count": len(results),
            "has_offers": any(self._is_true(r.get('metadata', {}).get('is_offer')) for r in results),
            "has_transfers": any(self._is_true(r.get('metadata', {}).get('has_transfer')) for r in results),
            "top_products": [
                {
                    'name': r.get('metadata', {}).get('PRODUCTO'),
                    'lab': r.get('metadata', {}).get('LABORATORIO'),
                    'score': r.get('total_score')
                }
                for r in results[:3]
            ]
        }
        
        print(json.dumps(final_structure, indent=4, ensure_ascii=False))
        
        print("\n📊 RESUMEN PARA MANAGER:")
        print(f"   ├─ Tiene resultados: {'✅' if results else '❌'}")
        print(f"   ├─ Filtros aplicados: {len(filters)}")
        print(f"   ├─ Exclusiones activas: {'✅' if filters.get('exclude_brands') else '❌'}")
        print(f"   ├─ Productos en oferta: {sum(1 for r in results if self._is_true(r.get('metadata', {}).get('is_offer')))}")
        print(f"   └─ Productos con transfer: {sum(1 for r in results if self._is_true(r.get('metadata', {}).get('has_transfer')))}")
    
    def _build_rag_context_preview(self, results: List[Dict]) -> str:
        """Simula cómo se construye el contexto RAG."""
        lines = ["--- INFORMACIÓN DEL CATÁLOGO ---"]
        
        for idx, result in enumerate(results, 1):
            meta = result.get('metadata', {})
            
            producto = meta.get('PRODUCTO') or meta.get('title', 'N/A')
            lab = meta.get('LABORATORIO', 'N/A')
            
            badges = []
            if self._is_true(meta.get('is_offer')):
                badges.append("🏷️ [EN OFERTA]")
            if self._is_true(meta.get('has_transfer')):
                badges.append("🎁 [TIENE TRANSFER]")
            
            header = f"[Producto #{idx}] {producto} | Lab: {lab} {' '.join(badges)}"
            lines.append(f"\n{header}")
            
            if meta.get('CATEGORIA'):
                lines.append(f"   > Categoría: {meta['CATEGORIA']}")
            
            if meta.get('CONCEPTO'):
                lines.append(f"   > Presentación: {meta['CONCEPTO']}")
            
            if meta.get('ESPECIE'):
                lines.append(f"   > Especie: {meta['ESPECIE']}")
            
            if meta.get('PRINCIPIO ACTIVO'):
                lines.append(f"   > Principio Activo: {meta['PRINCIPIO ACTIVO']}")
        
        lines.append("\n--- FIN DEL CATÁLOGO ---")
        return "\n".join(lines)
    
    def _is_true(self, value: Any) -> bool:
        """Helper para chequear booleans."""
        if isinstance(value, bool):
            return value
        return str(value).lower() in ('true', '1', 'yes', 'si')


# ═══════════════════════════════════════════════════════════════
# CASOS DE TEST COMPLETOS
# ═══════════════════════════════════════════════════════════════

def run_test_suite():
    """
    Ejecuta suite completa de tests con casos representativos.
    """
    
    visualizer = PipelineVisualizer()
    
    test_cases = [
        # Test 1: Búsqueda simple con peso
        {
            'name': 'Búsqueda Simple con Peso',
            'query': 'Power Gold de 20kg',
            'history': None
        },
        
        # Test 2: Búsqueda con exclusión
        {
            'name': 'Búsqueda con Exclusión',
            'query': 'antiparasitarios pero que no sea cidar',
            'history': None
        },
        
        # Test 3: Búsqueda de ofertas
        {
            'name': 'Búsqueda de Ofertas',
            'query': 'ofertas de Power',
            'history': None
        },
        
        # Test 4: Marca sin subfamilia
        {
            'name': 'Marca sin Subfamilia',
            'query': 'Cidar de 10kg',
            'history': None
        },
        
        # Test 5: Familia separada
        {
            'name': 'Familia Separada',
            'query': 'Nexgard Spectra',
            'history': None
        },
        
        # Test 6: Búsqueda por categoría con atributos
        {
            'name': 'Categoría + Atributos',
            'query': 'antiparasitario en pipeta para perros',
            'history': None
        },
        
        # Test 7: Transfers por laboratorio
        {
            'name': 'Transfers por Laboratorio',
            'query': 'transfers de Holliday',
            'history': None
        },
        
        # Test 8: Combinación de ofertas y transfers
        {
            'name': 'Ofertas Y Transfers',
            'query': 'promociones y transfers de antiparasitarios',
            'history': None
        },
        
        # Test 9: Refinamiento con historial
        {
            'name': 'Refinamiento con Historial',
            'query': 'pero en gotas',
            'history': [
                {
                    'entity': 'Power Gold',
                    'all_entities': [
                        {'type': 'PRODUCTO', 'value': 'Power Gold', 'score': 0.95}
                    ],
                    'intent': 'SEARCH',
                    'timestamp': '2024-01-01'
                }
            ]
        },
    ]
    
    results_summary = []
    
    for idx, test in enumerate(test_cases, 1):
        print("\n\n")
        print("█"*100)
        print(f"█  TEST CASE #{idx}: {test['name']}")
        print("█"*100)
        
        try:
            result = visualizer.test_complete_pipeline(
                test['query'],
                test.get('history')
            )
            
            results_summary.append({
                'test_name': test['name'],
                'query': test['query'],
                'status': '✅ PASS',
                'results_count': len(result['search_output'])
            })
        
        except Exception as e:
            print(f"\n❌ ERROR EN TEST: {e}")
            import traceback
            traceback.print_exc()
            
            results_summary.append({
                'test_name': test['name'],
                'query': test['query'],
                'status': f'❌ FAIL: {str(e)}',
                'results_count': 0
            })
    
    # Resumen final
    print("\n\n")
    print("█"*100)
    print("█  RESUMEN DE TESTS")
    print("█"*100)
    
    for idx, summary in enumerate(results_summary, 1):
        print(f"\n{idx}. {summary['test_name']}")
        print(f"   Query: '{summary['query']}'")
        print(f"   Status: {summary['status']}")
        print(f"   Resultados: {summary['results_count']}")
    
    # Estadísticas
    passed = sum(1 for s in results_summary if '✅' in s['status'])
    failed = len(results_summary) - passed
    
    print(f"\n{'─'*100}")
    print(f"Total Tests: {len(results_summary)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed/len(results_summary)*100):.1f}%")
    print("█"*100)


# ═══════════════════════════════════════════════════════════════
# TEST INDIVIDUAL CON MÁXIMO DETALLE
# ═══════════════════════════════════════════════════════════════

def test_single_query_detailed(query: str, history: List[Dict] = None):
    """
    Test ultra-detallado de una sola query.
    Ideal para debugging.
    """
    visualizer = PipelineVisualizer()
    
    print("\n" + "█"*100)
    print("█  TEST ULTRA-DETALLADO")
    print("█"*100)
    
    result = visualizer.test_complete_pipeline(query, history)
    
    # Guardar resultado en JSON para análisis
    output_file = f"test_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    serializable_result = {
        'query': query,
        'timestamp': datetime.now().isoformat(),
        'ner_output': {
            'intent': result['ner_output'].intent,
            'confidence': result['ner_output'].confidence,
            'entities': [
                {
                    'type': e.entity_type,
                    'value': e.entity_value,
                    'score': e.score,
                    'position': e.position
                }
                for e in result['ner_output'].all_entities
            ],
            'filters': result['ner_output'].filters
        },
        'optimizer_output': result['optimizer_output'],
        'search_output': [
            {
                'metadata': r.get('metadata', {}),
                'total_score': r.get('total_score'),
                'debug': r.get('_debug', {})
            }
            for r in result['search_output']
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultado guardado en: {output_file}")
    
    return result


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test individual desde línea de comandos
        query = " ".join(sys.argv[1:])
        test_single_query_detailed(query)
    else:
        # Suite completa
        print("🚀 Ejecutando Suite Completa de Tests...")
        run_test_suite()