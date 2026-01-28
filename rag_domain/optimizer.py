"""
OPTIMIZER V7.3 - FIXED
Filtrado de ruido del NER antes de pasar al LLM
"""
import json
import time
from typing import Dict, List, Optional, Set
from dataclasses import asdict

import requests

from ai_gateway.llm_client import LLMService
from rag_domain.ner_classifier import ClassificationResult


# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE FILTRADO
# ═══════════════════════════════════════════════════════════════

# Stop words que el NER detecta incorrectamente como entidades
STOP_WORDS = {
    'pero', 'aunque', 'sin', 'embargo', 'con', 'que', 'para',
    'de', 'en', 'y', 'o', 'a', 'la', 'el', 'los', 'las',
    'un', 'una', 'unos', 'unas', 'al', 'del'
}

# Palabras que son presentación, NO especie
PRESENTATION_KEYWORDS = {
    'gotas', 'pipeta', 'pipetas', 'comprimidos', 'tabletas',
    'inyectable', 'shampoo', 'spray', 'collar', 'difusor',
    'crema', 'gel', 'pomada', 'solucion', 'suspension'
}

# Rangos de peso estándar en veterinaria
COMMON_WEIGHT_RANGES = [
    (0, 4, 'muy_pequeno'),
    (4, 10, 'pequeno'),
    (10, 20, 'mediano'),
    (20, 40, 'grande'),
    (40, 60, 'muy_grande'),
    (60, 100, 'gigante')
]

# Productos sin subfamilia
PRODUCTS_WITHOUT_SUBFAMILY = {
    'bravecto', 'advocate', 'revolution', 'stronghold',
    'frontline', 'effitix', 'advantix', 'vectra',
    'comfortis', 'capstar', 'program', 'trifexis'
}

# Productos con familia separada
SEPARATE_FAMILIES = {
    'simparica trio': 'Producto independiente de Simparica',
    'nexgard spectra': 'Variante con espectro ampliado',
    'power gold': 'Variante Gold de Power',
    'power ultra': 'Variante Ultra de Power'
}

# ═══════════════════════════════════════════════════════════════
# KNOWLEDGE BASE VETERINARIA
# ═══════════════════════════════════════════════════════════════

VETERINARY_KNOWLEDGE = """
CRITICAL CLARIFICATIONS:

1. PRESENTATIONS (NOT drugs or species):
   - pipeta, gotas, comprimidos, tabletas, cápsulas
   - inyectable, shampoo, spray, collar, difusor
   - crema, gel, pomada, solución, suspensión
   → These go in "presentation" field, NEVER in "drug" or "species"

2. SPECIES (animals):
   - perro, gato, bovino, equino, ave, porcino, ovino
   → species field must be SINGLE STRING, not array
   → Format: "PERRO" not ["PERRO"]

3. WEIGHT RANGES:
   - "de 20kg" usually means range ending at 20 → infer 10-20 or 4-20
   - "hasta 10kg" means 0-10
   - "más de 40kg" means 40-999

4. EXCLUSIONS:
   - Only add to exclude_brands if user says "que no sea [SPECIFIC BRAND]"
   - Generic phrases like "que no sea X" or "menos X" are NOT exclusions
   - Example: "que no sea Bravecto" → exclude_brands: ["Bravecto"]
   - Example: "antiparasitarios pero que no sea bravecto" → exclude_brands: ["bravecto"]

5. BRAND NAMES:
   - Use the EXACT brand name from NER LABORATORIO entities
   - Do NOT add "LABORATORIO" prefix
   - Example: NER detects "HOLLIDAY" → brand: "HOLLIDAY" (not "LABORATORIO HOLLIDAY")
"""


class QueryOptimizer:
    """
    Query Optimizer con filtrado de ruido del NER.
    """
    
    def __init__(self):
        self.llm_client = LLMService()
    
    def optimize(
        self,
        query: str,
        classification: ClassificationResult,
        search_history: List[Dict] = None
    ) -> Dict:
        """
        Optimiza query con filtrado de ruido del NER.
        """
        
        print(f"\n🔍 [OPTIMIZER V7.3 FIXED] Processing: '{query}'")
        
        # ═══════════════════════════════════════════════════════════
        # PASO 1: FILTRAR ENTIDADES (NUEVO)
        # ═══════════════════════════════════════════════════════════
        filtered_entities = self._filter_noisy_entities(
            classification.all_entities,
            query
        )
        
        original_count = len(classification.all_entities)
        filtered_count = len(filtered_entities)
        removed = original_count - filtered_count
        
        if removed > 0:
            print(f"🧹 [FILTER] Removidas {removed} entidades ruidosas ({original_count} → {filtered_count})")
        
        # ═══════════════════════════════════════════════════════════
        # PASO 2: LLAMAR AL LLM con entidades filtradas
        # ═══════════════════════════════════════════════════════════
        llm_response = self._call_llm(
            query,
            filtered_entities,
            classification.filters,
            search_history or []
        )
        
        if not llm_response:
            return self._fallback_response(query, classification.intent)
        
        # ═══════════════════════════════════════════════════════════
        # PASO 3: POST-PROCESAMIENTO (validación adicional)
        # ═══════════════════════════════════════════════════════════
        optimized = self._post_process_response(llm_response, query)
        
        return optimized
    
    def _filter_noisy_entities(
        self,
        entities: List,
        query: str
    ) -> List:
        """
        Filtra entidades ruidosas del NER.
        
        Reglas:
        1. Remover stop words ("pero", "con", etc.)
        2. Remover presentaciones detectadas como especies
        3. Limitar productos en queries cortos (max 5)
        4. Remover duplicados
        """
        
        filtered = []
        seen_values = set()
        product_count = 0
        query_length = len(query.split())
        
        for entity in entities:
            if not isinstance(entity, dict):
                entity = asdict(entity)
            
            entity_type = entity.get('entity_type', '')
            entity_value = entity.get('entity_value', '').lower().strip()
            entity_score = entity.get('score', 0.0)
            
            # ═══════════════════════════════════════════════════════
            # REGLA 1: Stop words
            # ═══════════════════════════════════════════════════════
            if entity_value in STOP_WORDS:
                continue  # Skip "pero", "con", etc.
            
            # ═══════════════════════════════════════════════════════
            # REGLA 2: Presentación detectada como Especie
            # ═══════════════════════════════════════════════════════
            if entity_type == 'ESPECIE' and entity_value in PRESENTATION_KEYWORDS:
                # Es presentación, no especie
                # La agregamos como CONCEPTO en vez
                entity['entity_type'] = 'CONCEPTO'
                entity_type = 'CONCEPTO'
            
            # ═══════════════════════════════════════════════════════
            # REGLA 3: Limitar productos en queries cortos
            # ═══════════════════════════════════════════════════════
            if entity_type == 'PRODUCTO':
                product_count += 1
                
                # Si query tiene <5 palabras, limitar a 5 productos
                if query_length < 5 and product_count > 5:
                    continue  # Skip productos adicionales
                
                # Si query tiene >=5 palabras, limitar a 20 productos
                if query_length >= 5 and product_count > 20:
                    continue
            
            # ═══════════════════════════════════════════════════════
            # REGLA 4: Remover duplicados
            # ═══════════════════════════════════════════════════════
            if entity_value in seen_values:
                continue
            
            seen_values.add(entity_value)
            
            # ═══════════════════════════════════════════════════════
            # REGLA 5: Score mínimo (opcional)
            # ═══════════════════════════════════════════════════════
            # Si el score es muy bajo (<0.5), skip
            if entity_score < 0.5:
                continue
            
            # Entidad válida
            filtered.append(entity)
        
        return filtered
    
    def _call_llm(
    self,
    query: str,
    entities: List,
    filters: Dict,
    history: List
) -> Optional[Dict]:
        """
        Llama al LLM usando el servicio centralizado (LLMService).
        
        El servicio maneja:
        - Autenticación
        - Reintentos automáticos
        - Timeouts
        - Formato OpenAI-compatible
        """
        
        # ═══════════════════════════════════════════════════════════
        # PASO 1: Agrupar entidades por tipo
        # ═══════════════════════════════════════════════════════════
        entities_by_type = {}
        for e in entities:
            if not isinstance(e, dict):
                e = asdict(e)
            entity_type = e.get('entity_type', 'UNKNOWN')
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append({
                'value': e.get('entity_value', ''),
                'score': e.get('score', 0.0)
            })
        
        # ═══════════════════════════════════════════════════════════
        # PASO 2: Construir prompt del usuario
        # ═══════════════════════════════════════════════════════════
        user_prompt = self._build_prompt(
            query,
            entities_by_type,
            filters,
            history
        )
        
        # ═══════════════════════════════════════════════════════════
        # PASO 3: System prompt (MEJORADO con contexto veterinario)
        # ═══════════════════════════════════════════════════════════
        system_prompt = """You are a Query Optimizer API for a veterinary product search system.

    CRITICAL OUTPUT REQUIREMENTS:
    - Output ONLY valid JSON (no markdown, no preamble, no explanation)
    - Start with { and end with }
    - Use double quotes for strings
    - No trailing commas
    - No comments

    DOMAIN CONTEXT:
    - Veterinary products (antiparasitarios, alimentos, medicamentos)
    - Spanish language queries
    - Weight ranges in kg (perros/gatos: 2-60kg)
    - Presentations: pipeta, gotas, comprimidos, tabletas

    Your output will be parsed directly by json.loads(), so it must be perfect JSON."""
        
        # ═══════════════════════════════════════════════════════════
        # PASO 4: Llamar al servicio (con manejo de errores)
        # ═══════════════════════════════════════════════════════════
        try:
            print(f"📤 [OPTIMIZER] Sending to LLM via LLMService...")
            start_time = time.time()
            
            # CORRECCIÓN 2: Usamos self.llm_client (coincide con init)
            # LLMService.generate devuelve strings, no lanza requests exceptions
            output_text = self.llm_client.generate(system_prompt, user_prompt)
            
            elapsed = time.time() - start_time
            
            # Verificación básica de error devuelto como texto
            if not output_text or "Error:" in output_text[:10]:
                print(f"❌ [LLM] Service Error: {output_text}")
                return None
            
            print(f"✅ [LLM] Response received in {elapsed:.2f}s")
            
            # 5. Parsear (IGUAL)
            parsed = self._parse_llm_response(output_text)
            
            if parsed:
                # Validadores (IGUAL)
                if not parsed.get('search_input'): parsed['search_input'] = query
                if not parsed.get('search_filters'): parsed['search_filters'] = {}
                
                print(f"✅ [OPTIMIZER] Search term: '{parsed.get('search_input')}'")
                return parsed
            
            return None

        except Exception as e:
            # Capturamos cualquier error genérico (parsing, logic, etc)
            print(f"❌ [OPTIMIZER] Unexpected error: {e}")
            return None
            
        except json.JSONDecodeError as e:
            print(f"❌ [JSON] Parse error: {e}")
            print(f"   Position: line {e.lineno}, col {e.colno}")
            return None
            
        except requests.exceptions.Timeout:
            print(f"❌ [TIMEOUT] LLM service timed out")
            return None
            
        except requests.exceptions.ConnectionError as e:
            print(f"❌ [CONNECTION] Cannot reach LLM service: {e}")
            return None
            
        except Exception as e:
            print(f"❌ [OPTIMIZER] Unexpected error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _build_prompt(
        self,
        query: str,
        entities_by_type: Dict,
        filters: Dict,
        history: List
    ) -> str:
        """Construye prompt para el LLM con knowledge base."""
        
        # Generar hints contextuales
        hints = self._generate_contextual_hints(query, entities_by_type, history)
        
        # Extraer laboratorio del NER (si existe)
        lab_from_ner = None
        if 'LABORATORIO' in entities_by_type and entities_by_type['LABORATORIO']:
            lab_from_ner = entities_by_type['LABORATORIO'][0]['value']
        
        # Construir prompt
        prompt = f"""You are a veterinary product search query optimizer.

{VETERINARY_KNOWLEDGE}

USER QUERY: "{query}"

NER DETECTED ENTITIES (FILTERED):
{json.dumps(entities_by_type, indent=2, ensure_ascii=False)}

SEARCH HISTORY:
{json.dumps(history[-2:] if history else [], indent=2, ensure_ascii=False)}

CONTEXTUAL HINTS:
{chr(10).join(hints)}

YOUR TASK:
Generate optimized search parameters in JSON format:

{{
  "search_input": "<clean search term for vector search>",
  "search_filters": {{
    "brand": "<laboratorio name from NER or null>",
    "category": "<categoria or null>",
    "species": "<SINGLE species string or null>",
    "presentation": "<presentacion or null>",
    "drug": "<droga or null>",
    "weight_min": <number or null>,
    "weight_max": <number or null>,
    "is_offer": <boolean>,
    "is_transfer": <boolean>,
    "exclude_brands": [<list of SPECIFIC excluded brand names>]
  }},
  "intent": "SEARCH",
  "debug_analysis": {{
    "approved_entities": [<list of entity values used>],
    "excluded_brands": [<list of excluded brands>],
    "relevant_history_objects": [<list of history items used>]
  }}
}}

CRITICAL RULES:
1. search_input should be CLEAN (no weights, no presentations, no "ofertas")
2. species MUST be single string: "PERRO" not ["PERRO"]
3. If NER detected LABORATORIO, use its exact value in brand field
4. Presentations go in "presentation", NOT in "drug" or "species"
5. Only add to exclude_brands if user explicitly mentions specific brand to exclude
6. DO NOT infer species/presentation if not explicitly mentioned in query
7. If brand is unknown, use null (NOT "no detected" or "not found")
8. Respond ONLY with valid JSON, no markdown, no preamble

EXAMPLES OF INCORRECT INFERENCES (DO NOT DO THIS):
❌ Query: "Power Gold de 20kg" → species: "PERRO" (NOT mentioned!)
❌ Query: "Power Gold de 20kg" → presentation: "pipeta" (NOT mentioned!)
❌ Query: "Power Gold" → brand: "no detected" (use null instead!)
❌ Query: "antiparasitarios" → species: "PERRO" (user didn't specify!)

CORRECT APPROACH:
✅ Query: "Power Gold de 20kg" → ONLY: search_input="Power Gold", weight_min=10, weight_max=20
✅ Query: "antiparasitario en pipeta" → presentation="pipeta" (explicitly mentioned)
✅ Query: "Power para perros" → species="PERRO" (explicitly mentioned)
✅ Query: "ofertas de Power" → is_offer=true (explicitly mentioned)

{"IMPORTANT: Brand name detected by NER: " + lab_from_ner if lab_from_ner else ""}

RESPOND IN JSON:"""
        
        return prompt
    
    def _generate_contextual_hints(
        self,
        query: str,
        entities_by_type: Dict,
        history: List
    ) -> List[str]:
        """Genera hints para el LLM."""
        
        hints = []
        
        # Hint sobre laboratorio detectado
        if 'LABORATORIO' in entities_by_type and entities_by_type['LABORATORIO']:
            lab_name = entities_by_type['LABORATORIO'][0]['value']
            hints.append(
                f"🏭 LABORATORIO detected: '{lab_name}' → Use this EXACT name in brand field"
            )
        
        # Hint sobre productos sin subfamilia
        for product in entities_by_type.get('PRODUCTO', []):
            product_name = product['value'].lower()
            for no_subfamily in PRODUCTS_WITHOUT_SUBFAMILY:
                if no_subfamily in product_name:
                    hints.append(
                        f"⚠️ '{product['value']}' NO tiene subfamilias. "
                        f"Usa el nombre tal cual en search_input."
                    )
                    break
        
        # Hint sobre familias separadas
        for product in entities_by_type.get('PRODUCTO', []):
            product_name = product['value'].lower()
            for separate_family, note in SEPARATE_FAMILIES.items():
                if separate_family in product_name:
                    hints.append(f"ℹ️ '{product['value']}': {note}")
                    break
        
        # Hint sobre rangos de peso
        if any(char.isdigit() for char in query):
            hints.append(
                "💡 Si hay un peso único (ej: '20kg'), probablemente se refiere "
                "al rango estándar que termina en ese peso (ej: 10-20kg o 4-20kg)."
            )
        
        # Hint sobre ofertas/transfers
        if any(word in query.lower() for word in ['oferta', 'transfer', 'promo', 'descuento']):
            hints.append(
                "🏷️ No agregues 'oferta' o 'transfer' al search_input. "
                "Usa is_offer=true o is_transfer=true en los filtros."
            )
        
        # Hint sobre exclusiones (MÁS ESPECÍFICO)
        if 'no sea' in query.lower() or 'menos' in query.lower() or 'excepto' in query.lower():
            hints.append(
                "❌ EXCLUSIONS: Only add to exclude_brands if user mentions SPECIFIC brand name. "
                "Example: 'que no sea Bravecto' → exclude_brands: ['Bravecto']. "
                "Do NOT add generic phrases like 'que no sea X' or 'menos X'."
            )
        
        return hints
    
    def _parse_llm_response(self, output_text: str) -> Optional[Dict]:
        """Parsea respuesta del LLM."""
        
        # Limpiar markdown si existe
        output_clean = output_text.strip()
        
        if output_clean.startswith('```json'):
            output_clean = output_clean[7:]
        if output_clean.startswith('```'):
            output_clean = output_clean[3:]
        if output_clean.endswith('```'):
            output_clean = output_clean[:-3]
        
        output_clean = output_clean.strip()
        
        try:
            parsed = json.loads(output_clean)
            return parsed
        except json.JSONDecodeError as e:
            print(f"❌ [PARSER] JSON decode error: {e}")
            print(f"   Output was: {output_clean[:200]}")
            return None
    
    def _post_process_response(self, llm_response: Dict, query: str) -> Dict:
        """
        Post-procesamiento y validación robusta.
        
        FIXES V7.5:
        1. Brand: Validar "no detected" y valores inválidos
        2. Species: Solo mantener si está en el query
        3. Presentation: Solo mantener si está en el query
        4. Drug: Detectar presentaciones mal clasificadas
        5. Exclusiones: Filtrar alucinaciones
        """
        
        # Asegurar que search_input existe
        if not llm_response.get('search_input'):
            llm_response['search_input'] = query
        
        # Asegurar que search_filters existe
        if not llm_response.get('search_filters'):
            llm_response['search_filters'] = {}
        
        filters = llm_response['search_filters']
        query_lower = query.lower()
        
        # ═══════════════════════════════════════════════════════════
        # FIX #1: BRAND (validar alucinaciones y valores inválidos)
        # ═══════════════════════════════════════════════════════════
        if filters.get('brand'):
            brand = str(filters['brand']).strip()
            
            # Lista de valores inválidos
            INVALID_BRANDS = {
                'no detected', 'not detected', 'none', 'null', 'n/a',
                'no brand', 'unknown', 'not specified', 'not found',
                'no laboratorio', 'sin laboratorio', 'no lab'
            }
            
            if brand.lower() in INVALID_BRANDS or len(brand) < 3:
                filters['brand'] = None
                print(f"🔧 [POST-PROCESS] Removido brand inválido: '{brand}'")
            else:
                # Remover "LABORATORIO" prefix si existe
                if brand.upper().startswith('LABORATORIO '):
                    brand = brand[12:].strip()
                    filters['brand'] = brand
                    print(f"🔧 [POST-PROCESS] Limpiado brand: '{brand}'")
        
        # ═══════════════════════════════════════════════════════════
        # FIX #2: SPECIES (puede ser lista + validar si está en query)
        # ═══════════════════════════════════════════════════════════
        if filters.get('species'):
            species = filters['species']
            
            # Convertir lista a string
            if isinstance(species, list):
                species = species[0] if species else None
            
            if species:
                species_lower = str(species).lower()
                
                # Verificar si es presentación mal clasificada
                if species_lower in PRESENTATION_KEYWORDS:
                    if not filters.get('presentation'):
                        filters['presentation'] = species
                    filters['species'] = None
                    print(f"🔧 [POST-PROCESS] Movido '{species}' de species → presentation")
                else:
                    # Especies válidas
                    valid_species = {'perro', 'gato', 'canino', 'felino', 'bovino', 'equino', 'ave', 'porcino'}
                    
                    if species_lower in valid_species:
                        # Solo mantener si está mencionado en el query
                        if species_lower not in query_lower and 'perro' not in query_lower and 'gato' not in query_lower:
                            # El LLM infirió sin evidencia
                            filters['species'] = None
                            print(f"🔧 [POST-PROCESS] Removido species inferido: '{species}' (no está en query)")
                        else:
                            filters['species'] = species
                    else:
                        filters['species'] = None
        
        # ═══════════════════════════════════════════════════════════
        # FIX #3: PRESENTATION (validar si está en query)
        # ═══════════════════════════════════════════════════════════
        if filters.get('presentation'):
            pres = str(filters['presentation']).lower()
            
            # Solo mantener si está explícitamente en el query
            if pres not in query_lower:
                filters['presentation'] = None
                print(f"🔧 [POST-PROCESS] Removido presentation inferido: '{pres}' (no está en query)")
        
        # ═══════════════════════════════════════════════════════════
        # FIX #4: DRUG (puede ser presentación)
        # ═══════════════════════════════════════════════════════════
        if filters.get('drug'):
            drug = str(filters['drug']).lower()
            
            if drug in PRESENTATION_KEYWORDS:
                # Es presentación, no droga
                if not filters.get('presentation'):
                    filters['presentation'] = drug
                filters['drug'] = None
                print(f"🔧 [POST-PROCESS] Movido '{drug}' de drug → presentation")
        
        # ═══════════════════════════════════════════════════════════
        # FIX #5: EXCLUDE_BRANDS (filtrar alucinaciones)
        # ═══════════════════════════════════════════════════════════
        exclude_brands = filters.get('exclude_brands', [])
        
        if exclude_brands:
            # Patterns de alucinaciones comunes
            HALLUCINATION_PATTERNS = {
                'que no sea x', 'menos x', 'x', 'y',
                'specific brand', 'brand name', 'excluded brand',
                '[brand]', '<brand>', 'marca', 'lab', 'laboratorio'
            }
            
            # Filtrar alucinaciones
            original_count = len(exclude_brands)
            exclude_brands = [
                b for b in exclude_brands 
                if str(b).lower().strip() not in HALLUCINATION_PATTERNS
                and len(str(b).strip()) > 2  # Mínimo 3 caracteres
            ]
            
            if original_count != len(exclude_brands):
                removed = original_count - len(exclude_brands)
                print(f"🔧 [POST-PROCESS] Removidas {removed} exclusiones alucinadas")
            
            filters['exclude_brands'] = exclude_brands
        
        return llm_response
    
    def _fallback_response(self, query: str, intent: str) -> Dict:
        """Respuesta de fallback si el LLM falla."""
        
        return {
            'search_input': query,
            'search_filters': {},
            'intent': intent,
            'debug_analysis': {
                'approved_entities': [],
                'excluded_brands': [],
                'relevant_history_objects': []
            }
        }