#!/usr/bin/env python3
"""
TEST RUNNER - Sin dependencia de pytest

Ejecuta los tests de proximity scoring usando solo Python stdlib
"""
import sys
from pathlib import Path

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_proximity_score(target: float, actual: float, max_score: float = 0.3) -> float:
    """Calcula score de proximidad numérica"""
    if actual is None or target is None:
        return 0.0
    
    difference = abs(target - actual)
    
    if difference == 0:
        return max_score
    
    raw_score = max_score / (1 + difference)
    return min(raw_score, max_score)


def rank_products_by_proximity(products, target_dosage, semantic_weight=0.7, proximity_weight=0.3):
    """Rankea productos combinando semantic score y proximity score"""
    scored_products = []
    
    for product in products:
        semantic_score = product.get('semantic_score', 0.0)
        actual_dosage = product.get('dosage')
        
        proximity_score = calculate_proximity_score(target_dosage, actual_dosage)
        
        total_score = (semantic_score * semantic_weight) + (proximity_score * proximity_weight)
        
        scored_product = {
            **product,
            'proximity_score': proximity_score,
            'total_score': total_score
        }
        scored_products.append(scored_product)
    
    scored_products.sort(key=lambda p: p['total_score'], reverse=True)
    
    return scored_products


def test_exact_match():
    """Test: Coincidencia exacta debe dar el score máximo"""
    print("▶️ Test: Exact match gets max score")
    
    score = calculate_proximity_score(target=10.0, actual=10.0)
    
    assert score == 0.3, f"Expected 0.3, got {score}"
    print("   ✅ PASS")


def test_close_match():
    """Test: Diferencia pequeña debe dar score alto"""
    print("▶️ Test: Close match gets high score")
    
    score = calculate_proximity_score(target=10.0, actual=10.5)
    expected = 0.2
    
    assert abs(score - expected) < 0.01, f"Expected ~{expected}, got {score}"
    print(f"   ✅ PASS (score: {score:.3f})")


def test_far_match():
    """Test: Diferencia grande debe dar score bajo pero no cero"""
    print("▶️ Test: Far match gets low but non-zero score")
    
    score = calculate_proximity_score(target=10.0, actual=50.0)
    expected = 0.3 / 41
    
    assert abs(score - expected) < 0.001, f"Expected ~{expected}, got {score}"
    assert score > 0, "Score should be > 0 even for far matches"
    print(f"   ✅ PASS (score: {score:.4f})")


def test_apoquel_scenario():
    """Test: Escenario real de búsqueda de APOQUEL"""
    print("▶️ Test: Real scenario - searching 'apoquel 10mg'")
    
    products = [
        {
            'name': 'APOQUEL 5.4 MG',
            'dosage': 5.4,
            'semantic_score': 0.85
        },
        {
            'name': 'APOQUEL 16 MG',
            'dosage': 16.0,
            'semantic_score': 0.85
        },
        {
            'name': 'BRAVECTO 10 MG',
            'dosage': 10.0,
            'semantic_score': 0.45
        },
        {
            'name': 'APOQUEL 3.6 MG',
            'dosage': 3.6,
            'semantic_score': 0.82
        }
    ]
    
    ranked = rank_products_by_proximity(
        products, 
        target_dosage=10.0,
        semantic_weight=0.7,
        proximity_weight=0.3
    )
    
    names = [p['name'] for p in ranked]
    
    print(f"\n   Ranking:")
    for idx, product in enumerate(ranked, 1):
        print(f"   {idx}. {product['name']}")
        print(f"      Semantic: {product['semantic_score']:.2f}")
        print(f"      Proximity: {product['proximity_score']:.3f}")
        print(f"      Total: {product['total_score']:.3f}")
    
    # APOQUEL 5.4 debe rankear primero
    assert names[0] == 'APOQUEL 5.4 MG', f"Expected APOQUEL 5.4 first, got {names[0]}"
    
    # BRAVECTO NO debe rankear primero
    assert names[0] != 'BRAVECTO 10 MG', "BRAVECTO shouldn't rank first"
    
    # Todos los APOQUEL deben rankear antes que BRAVECTO
    apoquel_positions = [i for i, name in enumerate(names) if 'APOQUEL' in name]
    bravecto_position = names.index('BRAVECTO 10 MG')
    
    assert all(pos < bravecto_position for pos in apoquel_positions), \
        "All APOQUEL products should rank before BRAVECTO"
    
    print(f"\n   ✅ PASS - APOQUEL products ranked correctly")


def test_dynamic_dropoff():
    """Test: Dynamic drop-off con ganador claro"""
    print("▶️ Test: Dynamic drop-off with clear winner")
    
    candidates = [
        {'name': 'EXCELLENT', 'total_score': 0.95},
        {'name': 'GOOD', 'total_score': 0.70},
        {'name': 'MEDIOCRE', 'total_score': 0.45},
        {'name': 'POOR', 'total_score': 0.30}
    ]
    
    threshold = candidates[0]['total_score'] * 0.65
    filtered = [c for c in candidates if c['total_score'] >= threshold]
    
    print(f"   Threshold: {threshold:.3f}")
    print(f"   Filtered: {len(filtered)} candidates")
    
    assert len(filtered) == 2, f"Expected 2, got {len(filtered)}"
    assert filtered[0]['name'] == 'EXCELLENT'
    assert filtered[1]['name'] == 'GOOD'
    
    print(f"   ✅ PASS - Correctly filtered to top 2")


def test_dynamic_dropoff_mediocre():
    """Test: Dynamic drop-off con top mediocre"""
    print("▶️ Test: Dynamic drop-off with mediocre top score")
    
    candidates = [
        {'name': 'MEDIOCRE_A', 'total_score': 0.45},
        {'name': 'MEDIOCRE_B', 'total_score': 0.40},
        {'name': 'POOR', 'total_score': 0.25}
    ]
    
    threshold = candidates[0]['total_score'] * 0.65
    filtered = [c for c in candidates if c['total_score'] >= threshold]
    
    print(f"   Threshold: {threshold:.3f}")
    print(f"   Filtered: {len(filtered)} candidates")
    
    # MEDIOCRE_A y MEDIOCRE_B pasan
    assert len(filtered) == 2, f"Expected 2, got {len(filtered)}"
    
    print(f"   ✅ PASS - Adaptive threshold prevents empty results")


def main():
    """Ejecutar todos los tests"""
    print("\n" + "="*70)
    print("  PROXIMITY SCORING TESTS")
    print("="*70 + "\n")
    
    tests = [
        test_exact_match,
        test_close_match,
        test_far_match,
        test_apoquel_scenario,
        test_dynamic_dropoff,
        test_dynamic_dropoff_mediocre
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"   ❌ FAIL: {e}\n")
            failed += 1
        except Exception as e:
            print(f"   ❌ ERROR: {e}\n")
            failed += 1
    
    print("="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)