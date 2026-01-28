"""
TEST SUITE FOR QUERY OPTIMIZER V5 - REAL LLM (AUGMENTED)
Valida la integración NER + LLM Real.
Incluye casos base y nuevos casos basados en datos reales de producción.
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict

# --- FIX DE IMPORTS (Para ejecución directa con pytest) ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from rag_domain.optimizer import QueryOptimizer
except ImportError:
    try:
        from optimizer import QueryOptimizer
    except ImportError:
        raise ImportError("No se pudo encontrar 'optimizer.py'. Asegúrate de estar en la raíz del proyecto.")

# python -m pytest tests/test_ner.py -v -s

def test_ner_integration_real():
    print("\n" + "="*80)
    print("🚀 INICIANDO SUITE DE PRUEBAS - REAL LLM INTEGRATION (DATA ENRIQUECIDA)")
    print("="*80)
    
    # Instanciamos el optimizador
    try:
        opt = QueryOptimizer()
    except Exception as e:
        print(f"\n❌ ERROR AL INICIAR OPTIMIZER: {e}")
        return
    
    if not hasattr(opt, 'has_llm') or not opt.has_llm:
        print("\n❌ ERROR CRÍTICO: No se detectó el servicio LLM.")
        print("Asegúrate de configurar las variables de entorno para la API.")
        return

    print("✅ LLM Service Conectado. Ejecutando pruebas...\n")

    test_cases = [
        # ==========================================
        # GRUPO 1: CASOS DE LÓGICA (Validación Base)
        # ==========================================
        {
            "name": "Exclusión Semántica (No Credelio)",
            "query": "antiparasitario elanco pero no credelio",
            "desc": "El LLM debe eliminar 'Credelio' activamente.",
            "checks": {
                "filters.laboratorios": ["ELANCO"],
                "__exclude__": ["filters.target_products"] 
            }
        },
        {
            "name": "Recomendación (Garrapatas)",
            "query": "que tenes para garrapatas",
            "intent": "RECOMMENDATION",
            "desc": "Detectar 'garrapatas' como síntoma.",
            "checks": { "filters.symptoms": ["garrapatas"] }
        },

        # ==========================================
        # GRUPO 2: DATOS REALES DB (Nuevos Casos)
        # ==========================================
        
        # 1. CLINICO - ALGICAM (Antiinflamatorio)
        {
            "name": "Producto Específico + Droga (Algicam)",
            "query": "precio algicam pets meloxicam",
            "desc": "Validar cruce de Marca (Ceva/Algicam) y Droga.",
            "checks": {
                "filters.target_products": ["ALGICAM"], # Asume que 'ALGICAM' está en vademecum
                "filters.drogas": ["MELOXICAM"]
            }
        },

        # 2. ANTIPARASITARIO - ROCHY (Curabichera)
        {
            "name": "Categoría Específica (Rochy Curabichera)",
            "query": "rochy curabichera aerosol",
            "desc": "Validar producto y formato/categoría.",
            "checks": {
                "filters.target_products": ["ROCHY"],
                # 'CURABICHERA' suele ser CATEGORIA o ACCION según tu CSV
                # Verificamos que al menos uno de los dos lo capture
                "filters.categorias": ["CURABICHERA"] 
            }
        },

        # 3. ANTIARTROSICO - OL TRANS (Holliday)
        {
            "name": "Suplemento Complejo (Ol Trans Holliday)",
            "query": "ol trans polvo holliday",
            "desc": "Validar Laboratorio y Producto compuesto.",
            "checks": {
                "filters.laboratorios": ["HOLLIDAY"],
                "filters.target_products": ["OL TRANS"], # O "OL" y "TRANS" si el tokenizador separa
            }
        },

        # 4. OFERTA - VIRBAC FELIGEN (Biogenesis)
        {
            "name": "Oferta Vacunas (Virbac Feligen)",
            "query": "oferta feligen virbac",
            "desc": "Detectar intención comercial explícita y marca.",
            "checks": {
                "filters.is_offer": True,
                "filters.laboratorios": ["VIRBAC"],
                "filters.target_products": ["FELIGEN"]
            }
        },

        # 5. OFERTA + REGALO - POWER GOLD (Brouwer)
        {
            "name": "Promo con Regalo (Power Gold)",
            "query": "power gold con regalo de termo",
            "desc": "Detectar 'is_offer' por keyword 'regalo' y producto.",
            "checks": {
                "filters.is_offer": True,
                "filters.target_products": ["POWER"]
            }
        },

        # 6. TRANSFER - FIPRO Y PROTECH (Labyes)
        {
            "name": "Transfer/Bonificación (Labyes)",
            "query": "transfer fipro y protech labyes",
            "desc": "Detectar 'is_transfer' y múltiples productos.",
            "checks": {
                "filters.is_transfer": True,
                "filters.laboratorios": ["LABYES"],
                # Debería capturar ambos si están en el NER
                "filters.target_products": ["FIPRO", "PROTECH"] 
            }
        }
    ]

    passed_count = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"🔹 TEST {i}: {case['name']}")
        print(f"   Query: '{case['query']}'")
        
        start_time = time.time()
        try:
            result = opt.optimize(case['query'])
        except Exception as e:
            print(f"   ❌ ERROR EN EJECUCIÓN: {e}")
            continue

        duration = time.time() - start_time
        
        filters = result.get('search_filters', {})
        intent = result.get('intent', '')
        debug = result.get('debug_analysis', {})
        
        print(f"   ⏱️  Latencia LLM: {duration:.2f}s")
        print(f"   🔎 Decisión LLM: {json.dumps(debug.get('approved_entities', []), ensure_ascii=False)}")
        
        # Validaciones
        errors = [] 
        
        # 1. Validar Intent
        if 'intent' in case and intent != case['intent']:
            errors.append(f"❌ Intent incorrecto. Esperado: {case['intent']}, Actual: {intent}")

        # 2. Validar Filtros Positivos
        for key, expected_val in case.get('checks', {}).items():
            if key == "__exclude__": continue
            
            # Navegación segura por el diccionario
            parts = key.split('.')
            actual_val = filters
            for part in parts[1:]:
                if isinstance(actual_val, dict):
                    actual_val = actual_val.get(part)
                else:
                    actual_val = None
            
            # Comparación
            if isinstance(expected_val, list):
                if not actual_val:
                    # Fallo leve si el NER no tiene el dato cargado (ej: producto nuevo)
                    errors.append(f"⚠️ Falta filtro {key}. Esperado: {expected_val} (¿Está en vademecum.csv?)")
                else:
                    # Intersección flexible (case insensitive)
                    intersection = set([str(x).upper() for x in expected_val]) & set([str(x).upper() for x in actual_val])
                    if not intersection:
                        errors.append(f"❌ Valor {key} incorrecto. Esperado coincidencia con: {expected_val}, Actual: {actual_val}")
            else:
                if actual_val != expected_val:
                    errors.append(f"❌ Valor {key} incorrecto. Esperado: {expected_val}, Actual: {actual_val}")

        # 3. Validar Exclusiones
        if "__exclude__" in case.get('checks', {}):
            for key_to_exclude in case['checks']['__exclude__']:
                parts = key_to_exclude.split('.')
                actual_val = filters
                found = True
                for part in parts[1:]:
                    if isinstance(actual_val, dict):
                        actual_val = actual_val.get(part)
                    if actual_val is None:
                        found = False
                        break
                if found and actual_val:
                     errors.append(f"❌ Filtro {key_to_exclude} NO debería existir. Valor: {actual_val}")

        if not errors:
            print("   ✅ PASSED")
            passed_count += 1
        else:
            for e in errors:
                print(f"   {e}")
        print("-" * 50)

    print(f"\n🏁 RESULTADO FINAL: {passed_count}/{len(test_cases)} Pruebas exitosas.")

if __name__ == "__main__":
    test_ner_integration_real()