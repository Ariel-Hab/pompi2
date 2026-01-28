"""
TEST GENERAL INTEGRADO - Query Architect System
Valida el sistema completo con casos de uso reales del prompt original
"""
import json
from typing import List, Dict
from dataclasses import dataclass

# Imports simulados para el test (ajustar según tu estructura)
from rag_domain.ner_classifier import VeterinaryNERClassifier
from rag_domain.optimizer import QueryOptimizer
from rag_domain.search import VectorSearchService
from chat_domain.chat_history import ChatHistoryService


@dataclass
class TestCase:
    """Caso de test con validaciones"""
    name: str
    query: str
    expected_search_term: str
    expected_filters: Dict
    expected_exclusions: List[str]
    should_use_history: bool = False
    validation_rules: List[str] = None
    
    def __post_init__(self):
        if self.validation_rules is None:
            self.validation_rules = []


class QueryArchitectTestSuite:
    """Suite de tests para validar el Query Architect"""
    
    def __init__(self):
        self.ner = VeterinaryNERClassifier()
        self.optimizer = QueryOptimizer()
        self.search = VectorSearchService()
        self.passed = 0
        self.failed = 0
        self.results = []
    
    def run_all_tests(self):
        """Ejecuta todos los tests"""
        print("\n" + "="*80)
        print("🧪 QUERY ARCHITECT - TEST SUITE COMPLETO")
        print("="*80)
        
        test_cases = self._get_test_cases()
        
        for idx, test_case in enumerate(test_cases, 1):
            print(f"\n{'─'*80}")
            print(f"TEST {idx}/{len(test_cases)}: {test_case.name}")
            print(f"{'─'*80}")
            
            result = self._run_test(test_case)
            self.results.append(result)
            
            if result['passed']:
                self.passed += 1
                print(f"✅ PASS")
            else:
                self.failed += 1
                print(f"❌ FAIL")
                for error in result['errors']:
                    print(f"   • {error}")
        
        self._print_summary()
    
    def _get_test_cases(self) -> List[TestCase]:
        """Define los casos de test basados en el prompt original"""
        return [
            # ═══════════════════════════════════════════════════════
            # PROBLEMA 1: Over-collapsing
            # ═══════════════════════════════════════════════════════
            TestCase(
                name="Over-collapsing: Power Gold NO debe recortarse a Power",
                query="Power Gold de 10kg",
                expected_search_term="Power Gold",
                expected_filters={
                    'brand': 'Power',
                    'weight_min': 10
                },
                expected_exclusions=[],
                validation_rules=[
                    "search_term debe contener 'Power Gold' completo",
                    "NO debe ser solo 'Power'",
                    "weight_min debe ser 10"
                ]
            ),
            
            TestCase(
                name="Over-collapsing: Bravecto Ultra debe mantenerse completo",
                query="bravecto ultra para gatos",
                expected_search_term="Bravecto Ultra",
                expected_filters={
                    'species': 'GATO'
                },
                expected_exclusions=[],
                validation_rules=[
                    "search_term debe incluir 'Ultra'",
                    "species debe ser GATO"
                ]
            ),
            
            # ═══════════════════════════════════════════════════════
            # PROBLEMA 2: Falta de Atributos
            # ═══════════════════════════════════════════════════════
            TestCase(
                name="Atributos: 10kg debe aplicarse como filtro",
                query="antiparasitario de 10kg",
                expected_search_term="antiparasitario",
                expected_filters={
                    'weight_min': 10,
                    'weight_unit': 'kg'
                },
                expected_exclusions=[],
                validation_rules=[
                    "weight_min debe ser 10",
                    "weight_unit debe ser 'kg'",
                    "search_term NO debe contener '10kg'"
                ]
            ),
            
            TestCase(
                name="Atributos: Gotas debe aplicarse como filtro de presentación",
                query="power en gotas",
                expected_search_term="Power",
                expected_filters={
                    'presentation': 'GOTAS'
                },
                expected_exclusions=[],
                validation_rules=[
                    "presentation debe ser GOTAS",
                    "search_term NO debe contener 'gotas'"
                ]
            ),
            
            TestCase(
                name="Atributos: Pipeta debe aplicarse como filtro",
                query="antiparasitario en pipeta para perros",
                expected_search_term="antiparasitario",
                expected_filters={
                    'presentation': 'PIPETA',
                    'species': 'PERRO'
                },
                expected_exclusions=[],
                validation_rules=[
                    "presentation debe ser PIPETA",
                    "species debe ser PERRO"
                ]
            ),
            
            # ═══════════════════════════════════════════════════════
            # PROBLEMA 3: Lógica Negativa
            # ═══════════════════════════════════════════════════════
            TestCase(
                name="Lógica Negativa: 'No quiero Power' debe excluir Power",
                query="antiparasitarios pero que no sea power",
                expected_search_term="antiparasitarios",
                expected_filters={},
                expected_exclusions=['Power'],
                validation_rules=[
                    "exclude_brands debe contener 'Power'",
                    "search_term NO debe contener 'no' o 'que no sea'"
                ]
            ),
            
            TestCase(
                name="Lógica Negativa: 'Sin Bravecto' debe excluir Bravecto",
                query="antiparasitarios sin bravecto",
                expected_search_term="antiparasitarios",
                expected_filters={},
                expected_exclusions=['Bravecto'],
                validation_rules=[
                    "exclude_brands debe contener 'Bravecto'"
                ]
            ),
            
            TestCase(
                name="Lógica Negativa: 'Excepto X' debe excluir X",
                query="productos holliday excepto simparica",
                expected_search_term="Holliday",
                expected_filters={},
                expected_exclusions=['Simparica'],
                validation_rules=[
                    "exclude_brands debe contener 'Simparica'",
                    "search_term debe contener 'Holliday'"
                ]
            ),
            
            # ═══════════════════════════════════════════════════════
            # CASOS COMPLEJOS: Combinaciones
            # ═══════════════════════════════════════════════════════
            TestCase(
                name="Complejo: Marca + Peso + Exclusión",
                query="power gold de 10kg pero no el de bravecto",
                expected_search_term="Power Gold",
                expected_filters={
                    'brand': 'Power',
                    'weight_min': 10
                },
                expected_exclusions=['Bravecto'],
                validation_rules=[
                    "search_term debe ser 'Power Gold'",
                    "weight_min debe ser 10",
                    "exclude_brands debe contener 'Bravecto'"
                ]
            ),
            
            TestCase(
                name="Complejo: Categoría + Especie + Presentación",
                query="antiparasitario en pipeta para gatos",
                expected_search_term="antiparasitario",
                expected_filters={
                    'presentation': 'PIPETA',
                    'species': 'GATO'
                },
                expected_exclusions=[],
                validation_rules=[
                    "presentation debe ser PIPETA",
                    "species debe ser GATO"
                ]
            ),
            
            # ═══════════════════════════════════════════════════════
            # REFINAMIENTOS (requieren historial simulado)
            # ═══════════════════════════════════════════════════════
            TestCase(
                name="Refinamiento: 'pero en gotas' debe heredar búsqueda anterior",
                query="pero en gotas",
                expected_search_term="Power Gold",  # Del historial
                expected_filters={
                    'presentation': 'GOTAS'
                },
                expected_exclusions=[],
                should_use_history=True,
                validation_rules=[
                    "search_term debe heredarse del historial",
                    "presentation debe ser GOTAS"
                ]
            ),
        ]
    
    def _run_test(self, test_case: TestCase) -> Dict:
        """Ejecuta un caso de test"""
        try:
            print(f"Query: '{test_case.query}'")
            
            # 1. NER
            classification = self.ner.classify(test_case.query)
            print(f"   NER: {len(classification.all_entities)} candidatos")
            
            # 2. Simular historial si es necesario
            search_history = []
            if test_case.should_use_history:
                search_history = [
                    {
                        'entity': 'Power Gold',
                        'all_entities': [
                            {'type': 'PRODUCTO', 'value': 'Power Gold', 'score': 0.95}
                        ],
                        'intent': 'SEARCH',
                        'timestamp': '2024-01-01'
                    }
                ]
            
            # 3. Optimizer (Query Architect)
            optimized = self.optimizer.optimize(
                test_case.query,
                classification,
                search_history=search_history
            )
            
            search_term = optimized.get('search_input', '')
            filters = optimized.get('search_filters', {})
            exclusions = filters.get('exclude_brands', [])
            
            print(f"   Architect:")
            print(f"     Search Term: '{search_term}'")
            print(f"     Filters: {list(filters.keys())}")
            if exclusions:
                print(f"     Exclusions: {exclusions}")
            
            # 4. Validaciones
            errors = []
            
            # Validar search_term
            if test_case.expected_search_term:
                expected_lower = test_case.expected_search_term.lower()
                actual_lower = search_term.lower()
                
                if expected_lower not in actual_lower:
                    errors.append(
                        f"Search term incorrecto. "
                        f"Esperado: '{test_case.expected_search_term}', "
                        f"Obtenido: '{search_term}'"
                    )
            
            # Validar filtros
            for key, expected_value in test_case.expected_filters.items():
                actual_value = filters.get(key)
                
                if isinstance(expected_value, str):
                    if not actual_value or expected_value.lower() not in str(actual_value).lower():
                        errors.append(
                            f"Filtro '{key}' incorrecto. "
                            f"Esperado: '{expected_value}', "
                            f"Obtenido: '{actual_value}'"
                        )
                else:
                    if actual_value != expected_value:
                        errors.append(
                            f"Filtro '{key}' incorrecto. "
                            f"Esperado: {expected_value}, "
                            f"Obtenido: {actual_value}"
                        )
            
            # Validar exclusiones
            for expected_exclusion in test_case.expected_exclusions:
                found = any(
                    expected_exclusion.lower() in excl.lower() 
                    for excl in exclusions
                )
                if not found:
                    errors.append(
                        f"Exclusión faltante: '{expected_exclusion}' "
                        f"no está en {exclusions}"
                    )
            
            # Validar reglas adicionales
            for rule in test_case.validation_rules:
                if not self._validate_rule(rule, search_term, filters, exclusions):
                    errors.append(f"Regla no cumplida: {rule}")
            
            return {
                'test_case': test_case.name,
                'query': test_case.query,
                'passed': len(errors) == 0,
                'errors': errors,
                'results': {
                    'search_term': search_term,
                    'filters': filters,
                    'exclusions': exclusions
                }
            }
            
        except Exception as e:
            return {
                'test_case': test_case.name,
                'query': test_case.query,
                'passed': False,
                'errors': [f"Exception: {str(e)}"],
                'results': {}
            }
    
    def _validate_rule(self, rule: str, search_term: str, filters: Dict, exclusions: List[str]) -> bool:
        """Valida una regla específica"""
        rule_lower = rule.lower()
        
        # Regla: NO debe contener X
        if "no debe contener" in rule_lower or "no debe ser" in rule_lower:
            forbidden_words = ['no', 'kg', 'g', 'ml', 'gotas', 'pipeta', 'comprimidos']
            for word in forbidden_words:
                if word in rule_lower and word in search_term.lower():
                    return False
        
        # Regla: debe incluir X
        if "debe incluir" in rule_lower or "debe contener" in rule_lower:
            # Extraer qué debe incluir
            if "'" in rule:
                required = rule.split("'")[1]
                if required.lower() not in search_term.lower():
                    return False
        
        return True
    
    def _print_summary(self):
        """Imprime resumen de tests"""
        print("\n" + "="*80)
        print("📊 RESUMEN DE TESTS")
        print("="*80)
        
        total = self.passed + self.failed
        print(f"\nTotal: {total} tests")
        print(f"✅ Pasaron: {self.passed}")
        print(f"❌ Fallaron: {self.failed}")
        
        if self.failed > 0:
            print(f"\n🔍 TESTS FALLIDOS:")
            for result in self.results:
                if not result['passed']:
                    print(f"\n  • {result['test_case']}")
                    print(f"    Query: '{result['query']}'")
                    for error in result['errors']:
                        print(f"      - {error}")
        
        print("\n" + "="*80)
        
        if self.failed == 0:
            print("🎉 ¡TODOS LOS TESTS PASARON!")
        else:
            print(f"⚠️ {self.failed} tests fallaron. Revisar implementación.")
        
        print("="*80 + "\n")


# ═══════════════════════════════════════════════════════════════
# TEST DE BÚSQUEDA CON EXCLUSIONES
# ═══════════════════════════════════════════════════════════════

def test_search_with_exclusions():
    """Test específico de búsqueda con exclusiones"""
    print("\n" + "="*80)
    print("🔍 TEST: BÚSQUEDA CON EXCLUSIONES")
    print("="*80)
    
    search = VectorSearchService()
    
    # Test 1: Búsqueda normal (sin exclusiones)
    print("\n📝 Test 1: Búsqueda sin exclusiones")
    optimized_data = {
        'search_input': 'antiparasitario',
        'search_filters': {
            'category': 'ANTIPARASITARIO',
            'species': 'PERRO'
        }
    }
    
    results_without_exclusion = search.search_with_context(optimized_data, top_k=10)
    print(f"   Resultados: {len(results_without_exclusion)}")
    
    # Contar cuántos son Bravecto
    bravecto_count = sum(
        1 for r in results_without_exclusion 
        if 'bravecto' in str(r.get('metadata', {}).get('PRODUCTO', '')).lower()
    )
    print(f"   Productos Bravecto: {bravecto_count}")
    
    # Test 2: Misma búsqueda CON exclusión de Bravecto
    print("\n📝 Test 2: Misma búsqueda excluyendo Bravecto")
    optimized_data_with_exclusion = {
        'search_input': 'antiparasitario',
        'search_filters': {
            'category': 'ANTIPARASITARIO',
            'species': 'PERRO',
            'exclude_brands': ['Bravecto']
        }
    }
    
    results_with_exclusion = search.search_with_context(optimized_data_with_exclusion, top_k=10)
    print(f"   Resultados: {len(results_with_exclusion)}")
    
    # Verificar que NO hay Bravecto
    bravecto_found = [
        r.get('metadata', {}).get('PRODUCTO', 'N/A')
        for r in results_with_exclusion 
        if 'bravecto' in str(r.get('metadata', {}).get('PRODUCTO', '')).lower()
    ]
    
    if bravecto_found:
        print(f"   ❌ FALLO: Se encontraron productos Bravecto: {bravecto_found}")
    else:
        print(f"   ✅ PASS: No se encontraron productos Bravecto")
    
    print("\n" + "="*80)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Ejecutar suite completa
    suite = QueryArchitectTestSuite()
    suite.run_all_tests()
    
    # Test específico de búsqueda
    test_search_with_exclusions()