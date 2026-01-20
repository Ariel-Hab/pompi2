"""
NER Classifier v2 - Con helper de conversión a formato Optimizer
Mantiene toda la funcionalidad original + agrega interoperabilidad
"""
import re
import csv
import json
import difflib
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field


@dataclass
class EntityMatch:
    entity_type: str 
    entity_value: str
    position: int
    length: int
    original_text: str = ""

    @property
    def match_diff(self) -> int:
        """Calcula la penalización por diferencia de longitud (Ruido vs Señal)."""
        if not self.original_text: 
            return 0
        return len(self.original_text) - self.length


@dataclass
class ClassificationResult:
    entity_type: Optional[str]
    entity_value: Optional[str]
    all_entities: List[EntityMatch] = field(default_factory=list)
    filters: Dict = field(default_factory=dict)
    intent: str = "SEARCH"
    confidence: float = 0.5


# En ner_classifier.py

def classification_to_optimizer_format(classification: ClassificationResult) -> Dict:
    """
    Convierte a formato Optimizer usando TODAS las entidades.
    """
    # 1. Agrupar por tipos
    labs = [e.entity_value for e in classification.all_entities if e.entity_type == "LABORATORIO"]
    cats = [e.entity_value for e in classification.all_entities if e.entity_type == "CATEGORIA"]
    actions = [e.entity_value for e in classification.all_entities if e.entity_type == "ACCION"]
    products = [e.entity_value for e in classification.all_entities if e.entity_type == "PRODUCTO"]
    species = [e.entity_value for e in classification.all_entities if e.entity_type == "ESPECIE"]

    # 2. Construir detalles combinados (Contexto para RAG/LLM)
    context_parts = []
    context_parts.extend(products)
    context_parts.extend(labs)
    context_parts.extend(cats)
    context_parts.extend(actions)
    
    details_text = " ".join(context_parts)

    # 3. Inyectar listas completas en filtros
    classification.filters['target_products'] = products
    
    return {
        "intent": classification.intent,
        # ❌ ELIMINAR: "main_entity": "",
        "search_input": details_text,  # ✅ Usamos el details_text como input
        "search_filters": classification.filters,
        "parsed_metadata": {
            "detected_labs": labs,
            "detected_products": products,
            "detected_categories": cats,
            "detected_species": species,
            "dosage_value": classification.filters.get('dosage_value'),
            "dosage_unit": classification.filters.get('dosage_unit'),
            "presentation_normalized": classification.filters.get('presentation'),
        }
    }


class VeterinaryNERClassifier:
    """
    Clasificador NER Híbrido para E-commerce Veterinario.
    
    ARQUITECTURA:
    1. Normalización: Convierte todo a mayúsculas internamente y remueve acentos.
    2. Sanitización: Limpia el CSV cargado eliminando 'ruido' y contaminación cruzada.
    3. Estrategias de Búsqueda:
       a. Exact Match: Busca substrings exactos. Ideal para entidades compuestas.
       b. Fuzzy Match: Usa difflib para corregir typos y plurales no exactos.
       c. Heurística de Productos: Detecta productos por raíz para evitar falsos negativos.
    4. Ranking y Deduplicación:
       Ordena candidatos por Posición > Calidad del Match > Longitud > Prioridad.
       Elimina duplicados conservando solo el mejor candidato para cada entidad detectada.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            self.data_path = project_root / "data"
        else:
            self.data_path = Path(data_dir)

        # Palabras a ignorar
        self.ignored_terms = {
            'para', 'con', 'sin', 'los', 'las', 'una', 'unos', 'del', 'por',
            'uso', 'veterinario', 'envase', 'caja', 'peso', 'vivo', 'kpv', 'kilos',
            'comprimido', 'comprimidos', 'ml', 'mg', 'gr', 'accion', 'terapeutica',
            'tratamiento', 'enfermedades', 'amplio', 'espectro', 'via', 'oral',
            'producto', 'productos', 'lista', 'precio', 'venta'
        }
        
        self.stop_phrases = [
            "que productos", "quien vende", "quien tiene", "que tiene", "busco", 
            "necesito", "precio de", "tenes", "tienen", "tiene", # <--- AGREGADO AQUI
            "info de", "productos de", "informacion de", 
            "dame", "quiero", "me das", "consulta sobre", "decime", "contame", "mostrame"
        ]
        
        self.smalltalk_keywords = {
            'hola', 'hey', 'buenos', 'dias', 'tardes', 'como', 'andas', 
            'estas', 'gracias', 'chau', 'adios'
        }
        self.recommendation_keywords = {
            'recomiend', 'sugier', 'que usar', 'que darle', 'que le doy', 
            'ayuda con', 'necesito para', 'tiene para', 'sirve para'
        }
        self.offer_keywords = {
            'oferta', 'off', 'desc', 'promo', 'descuento', 'rebaja', 'liquidacion'
        }
        self.transfer_keywords = {
            'transfer', 'bonif', 'regalo', 'regla', 'combo', 'pack', 'bm'
        }
        
        # --- CARGA DE DATOS ---
        vademecum_path = self.data_path / "vademecum.csv"
        enterprises_path = self.data_path / "enterprises.csv"

        # Tokenize=True para Laboratorios ("Bayer" de "Bayer S.A.")
        self.laboratorios = self._load_known_entities(
            enterprises_path, "title", tokenize=True
        )
        self.productos = self._load_known_entities(vademecum_path, "PRODUCTO")
        self.drogas = self._load_known_entities(
            vademecum_path, "PRINCIPIO ACTIVO", 
            split_chars=[',', '+', '/', ' y ', ' Y '], 
            tokenize=True
        )
        self.categorias = self._load_known_entities(vademecum_path, "CATEGORIA", tokenize=True)
        self.acciones = self._load_known_entities(
            vademecum_path, "ACCION TERAPEUTICA", 
            split_chars=[',', '-', '/', '.', ':'], # Agregado ':' aquí
            tokenize=True
        )
        self.conceptos = self._load_known_entities(
            vademecum_path, "CONCEPTO", 
            split_chars=['/'], 
            tokenize=False
        )
        self.especies = self._load_known_entities(
            vademecum_path, "ESPECIE", 
            split_chars=[',', ';', '/'], 
            tokenize=True
        )
        
        self._enrich_species_vocabulary()
        self._sanitize_entities() 

    def _enrich_species_vocabulary(self):
        """Enriquece vocabulario de especies con sinónimos y plurales"""
        synonyms = {
            'canino', 'caninos', 'felino', 'felinos', 'cachorro', 'cachorros', 
            'gatito', 'gatitos', 'equino', 'equinos', 'bovino', 'bovinos', 
            'perro', 'perros', 'gato', 'gatos',
            'pulga', 'pulgas', 'garrapata', 'garrapatas'
        }
        self.especies.update({s.upper() for s in synonyms})
        
        extras = set()
        for sp in self.especies:
            sp_lower = sp.lower()
            if sp_lower.endswith(('a', 'e', 'i', 'o', 'u')): 
                extras.add(sp + "S")
            elif sp_lower[-1].isalpha(): 
                extras.add(sp + "ES")
        self.especies.update(extras)

    def _sanitize_entities(self):
        """Limpia entidades para evitar contaminación cruzada"""
        stop_words_veterinaria = {
            'PARA', 'USO', 'VETERINARIO', 'DEL', 'LOS', 'LAS', 'CON', 'SIN'
        }
        
        # Limpiar Drogas
        self.drogas = {
            d for d in self.drogas 
            if d not in self.especies and d not in stop_words_veterinaria
        }
        
        # Limpiar Acciones
        self.acciones = {
            a for a in self.acciones 
            if a not in self.especies and a not in stop_words_veterinaria
        }

        # Asegurar manuales
        manual_concepts = {
            'COMPRIMIDO', 'COMPRIMIDOS', 'JARABE', 'GOTAS', 'PIPETA', 
            'COLLAR', 'COLLARES', 'VACUNA', 'VACUNAS'
        }
        self.conceptos.update(manual_concepts)
        
        manual_cats = {'CLINICO', 'FARMACIA', 'ALIMENTO', 'ACCESORIO'}
        self.categorias.update(manual_cats)

    def classify(self, query: str) -> ClassificationResult:
        """
        Clasifica query detectando TODAS las entidades sin jerarquía excluyente.
        """
        query_norm = self._normalize_text(query)
        query_clean = self._clean_query(query_norm)
        intent = self._detect_intent(query_norm, query_clean)
        
        # Detección de Smalltalk puro
        if intent == "SMALLTALK" and not self._mentions_products(query_clean):
            return ClassificationResult(
                None, None, [], {}, "SMALLTALK", 1.0
            )
        
        # 1. Encontrar todas las entidades
        all_entities = self._find_all_entities(query_clean, query_norm)
        
        # 2. Extraer filtros numéricos
        filters = self._extract_filters(query_norm)
        
        # 3. Calcular confianza
        confidence = self._calculate_confidence(all_entities, filters)
        
        # --- CAMBIO: YA NO ELEGIMOS UNA "ENTITY_VALUE" ÚNICA ---
        # Dejamos entity_type y entity_value en None o vacíos.
        # La verdad está en 'all_entities'.
        
        return ClassificationResult(
            entity_type=None,   # Antes: primary_entity["type"]
            entity_value=None,  # Antes: primary_entity["value"]
            all_entities=all_entities,
            filters=filters,
            intent=intent,
            confidence=confidence
        )
    
    def _find_all_entities(
        self, query_clean: str, query_norm: str
    ) -> List[EntityMatch]:
        """
        Encuentra TODAS las entidades en el query usando estrategias híbridas.
        
        ESTRATEGIAS:
        1. Exact Match - Substrings exactos con word boundaries
        2. Fuzzy Match - difflib para typos y variaciones
        3. Product Heuristic - Matching por raíz de producto
        """
        matches = []
        query_lower = query_clean.lower()
        
        # --- ESTRATEGIA 1: BÚSQUEDA EXACTA ---
        def search_exact(entity_set, type_name, min_len=3):
            for item in entity_set:
                item_lower = item.lower()
                if len(item_lower) < min_len: 
                    continue     
                if item_lower not in query_lower: 
                    continue

                start_index = 0
                while True:
                    try:
                        pos = query_lower.index(item_lower, start_index)
                    except ValueError: 
                        break 
                    
                    # Word boundary check
                    is_start_ok = (pos == 0) or (not query_lower[pos-1].isalnum())
                    is_end_ok = (
                        pos + len(item_lower) == len(query_lower)
                    ) or (
                        not query_lower[pos + len(item_lower)].isalnum()
                    )
                    
                    if is_start_ok and is_end_ok:
                        if not self._is_hard_overlapping(
                            pos, len(item_lower), matches
                        ):
                            matches.append(EntityMatch(
                                type_name, item, pos, len(item_lower), 
                                original_text=item
                            ))
                    start_index = pos + 1

        search_exact(self.laboratorios, "LABORATORIO")
        search_exact(self.drogas, "DROGA")
        search_exact(self.acciones, "ACCION")
        search_exact(self.categorias, "CATEGORIA")
        search_exact(self.conceptos, "CONCEPTO")
        search_exact(self.especies, "ESPECIE")

        # --- ESTRATEGIA 2: FUZZY MATCHING ---
        query_words = query_clean.split()
        
        fuzzy_targets = {
            "CATEGORIA": self.categorias,
            "ESPECIE": self.especies,
            "CONCEPTO": self.conceptos,
            "ACCION": self.acciones,
            "LABORATORIO": self.laboratorios, 
            "DROGA": self.drogas 
        }

        for i, word in enumerate(query_words):
            if len(word) < 4: 
                continue
            
            # --- AGREGAR ESTAS 2 LÍNEAS AQUÍ ---
            # Si la palabra es ruido (ej: "oral", "peso"), no intentar corregirla
            if word.lower() in self.ignored_terms:
                continue
            # -----------------------------------
            
            word_pos = query_lower.find(word.lower())
            if word_pos != -1 and self._is_hard_overlapping(
                word_pos, len(word), matches
            ):
                continue

            for type_name, entity_set in fuzzy_targets.items():
                candidates = difflib.get_close_matches(
                    word.upper(), entity_set, n=1, cutoff=0.85
                )
                if candidates:
                    match_val = candidates[0]
                    if word_pos != -1 and not self._is_hard_overlapping(
                        word_pos, len(word), matches
                    ):
                        matches.append(EntityMatch(
                            type_name, match_val, word_pos, len(word), 
                            original_text=match_val
                        ))
                        break

        # --- ESTRATEGIA 3: PRODUCTOS (Raíz + Fuzzy) ---
        for prod in self.productos:
            prod_lower = prod.lower()
            if prod_lower in query_lower:
                matches.append(EntityMatch(
                    "PRODUCTO", prod, query_lower.index(prod_lower), 
                    len(prod), original_text=prod
                ))
            else:
                parts = prod_lower.split()
                if not parts: 
                    continue
                
                # Match por raíz
                root = parts[0]
                if len(parts) > 1 and len(parts[0]) < 4: 
                    root = f"{parts[0]} {parts[1]}"
                
                if len(root) < 3 or root in self.ignored_terms: 
                    continue

                if root in query_lower:
                    idx = query_lower.index(root)
                    if not self._is_hard_overlapping(idx, len(root), matches):
                        matches.append(EntityMatch(
                            "PRODUCTO", prod, idx, len(root), 
                            original_text=prod
                        ))
                else:
                    # Fuzzy con raíz
                    candidates = difflib.get_close_matches(
                        root.upper(), 
                        [w.upper() for w in query_words], 
                        n=1, cutoff=0.85
                    )
                    if candidates:
                        word_matched = candidates[0].lower()
                        if word_matched in query_lower:
                            idx = query_lower.index(word_matched)
                            if not self._is_hard_overlapping(
                                idx, len(word_matched), matches
                            ):
                                matches.append(EntityMatch(
                                    "PRODUCTO", prod, idx, len(word_matched), 
                                    original_text=prod
                                ))

        # --- RANKING ---
        matches.sort(key=lambda m: (
            m.match_diff,                                   # 1. Calidad (Exacto gana a Fuzzy)
            self._get_entity_type_priority(m.entity_type),  # 2. Importancia (Producto > Accion)
            -m.length,                                      # 3. Longitud (Más largo mejor)
            m.position                                      # 4. Posición (Último recurso)
        ))
        
        # --- DEDUPLICACIÓN ---
        final_matches = []
        seen_entities = set()

        for m in matches:
            key = (m.entity_type, m.entity_value)
            if key not in seen_entities:
                seen_entities.add(key)
                final_matches.append(m)

        return final_matches
    
    def _is_hard_overlapping(
        self, position: int, length: int, existing_matches: List[EntityMatch]
    ) -> bool:
        """
        Detecta overlap PARCIAL con entidades existentes.
        Permite el mismo span exacto (multi-label).
        """
        new_end = position + length
        for match in existing_matches:
            # Mismo span exacto = OK (multi-label)
            if match.position == position and match.length == length:
                continue 

            existing_end = match.position + match.length
            # Chequeo de cruce
            if not (new_end <= match.position or position >= existing_end):
                return True
        return False
    
    def _get_entity_type_priority(self, entity_type: str) -> int:
        """Prioridades para ranking de entidades"""
        priorities = {
            "PRODUCTO": 1,
            "LABORATORIO": 2,
            "CATEGORIA": 3,
            "DROGA": 4,
            "ACCION": 5,
            "CONCEPTO": 6,
            "ESPECIE": 7
        }
        return priorities.get(entity_type, 99)
    
    # def _select_primary_entity(
    #     self, all_entities: List[EntityMatch]
    # ) -> Optional[dict]:
    #     """Selecciona entidad principal (primero en ranking)"""
    #     if not all_entities: 
    #         return None
    #     best = all_entities[0]
    #     return {"type": best.entity_type, "value": best.entity_value}

    def _load_known_entities(
        self, 
        csv_path: Path, 
        column: str, 
        split_chars: List[str] = None, 
        tokenize: bool = False
    ) -> Set[str]:
        """Carga entidades conocidas desde CSV con limpieza profunda de puntuación"""
        entities = set()
        if not csv_path.exists(): 
            return entities
            
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_val = row.get(column, '').strip()
                if not raw_val or raw_val in ['0', 'N/A', 'None']: 
                    continue
                
                values_to_process = [raw_val]
                if split_chars:
                    temp = [raw_val]
                    for char in split_chars:
                        new_temp = []
                        for v in temp: 
                            new_temp.extend(v.split(char))
                        temp = new_temp
                    values_to_process = [v.strip() for v in temp if v.strip()]
                
                for val in values_to_process:
                    # 1. Limpieza básica
                    val_clean = re.sub(r'\(.*?\)', '', val).strip()
                    # 2. Remover puntuación pegada al final (Clave para 'PARA:' y 'ORAL:')
                    val_clean = val_clean.strip(".:;,")

                    if val_clean.lower() in self.ignored_terms: 
                        continue

                    if len(val_clean) > 2:
                        entities.add(val_clean.upper()) 
                        if tokenize:
                            words = val_clean.split()
                            if len(words) > 1:
                                for w in words:
                                    # 3. Limpiar también los tokens individuales
                                    w_clean = w.strip(".:;,")
                                    if (len(w_clean) > 3 and 
                                        w_clean.lower() not in self.ignored_terms):
                                        entities.add(w_clean.upper())
        return entities

    def _normalize_text(self, text: str) -> str:
        """Normaliza texto: lowercase, sin acentos, espacios únicos"""
        text = text.lower().strip()
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore').decode("utf-8")
        return re.sub(r'\s+', ' ', text)
    
    def _clean_query(self, text: str) -> str:
        """Limpia query de stop phrases y caracteres especiales"""
        cleaned = text
        for phrase in self.stop_phrases: 
            cleaned = cleaned.replace(phrase, "")
        return re.sub(r'[?¿!¡]', '', cleaned).strip()

    def _detect_intent(self, query_norm: str, query_clean: str) -> str:
        """Detecta intención basándose en keywords"""
        if (any(kw in query_norm for kw in self.smalltalk_keywords) and 
            not self._mentions_products(query_clean)):
            return "SMALLTALK"
        if any(kw in query_norm for kw in self.recommendation_keywords):
            return "RECOMMENDATION"
        return "SEARCH"

    def _extract_filters(self, query: str) -> dict:
        """Extrae filtros numéricos y booleanos del query"""
        filters = {}
        
        # Dosage
        match_dose = re.search(r'(\d+(?:[,\.]\d+)?)\s*(mg|ml|gr?)', query)
        if match_dose:
            filters['dosage_value'] = float(
                match_dose.group(1).replace(',', '.')
            )
            filters['dosage_unit'] = match_dose.group(2)
        
        # Presentation
        pres_map = {
            'comprimido': 'comprimidos', 
            'tableta': 'comprimidos', 
            'inyectable': 'inyectable'
        }
        for k, v in pres_map.items():
            if k in query: 
                filters['presentation'] = v
            
        # Flags
        if any(kw in query for kw in self.offer_keywords): 
            filters['is_offer'] = True
        if any(kw in query for kw in self.transfer_keywords): 
            filters['is_transfer'] = True
        
        return filters

    def _mentions_products(self, query: str) -> bool:
        """
        Detecta si query menciona productos reales.
        Retorna True solo si quedan palabras significativas después de quitar saludos.
        """
        # Tokenizamos y limpiamos
        words = query.lower().split()
        
        # Filtramos palabras que sean saludos o stopwords comunes
        meaningful_words = [
            w for w in words 
            if w not in self.smalltalk_keywords 
            and w not in self.stop_phrases
            and len(w) > 2 # Ignoramos "y", "el", etc.
        ]
        
        # Si queda algo (ej: "bravecto"), es búsqueda. 
        # Si no queda nada (ej: "hola buenos dias"), es smalltalk.
        return len(meaningful_words) > 0
    
    def _calculate_confidence(self, entities, filters):
        """Calcula score de confianza basado en cantidad de detecciones"""
        return min(
            0.5 + len(entities) * 0.2 + (0.1 if filters else 0), 
            1.0
        )
if __name__ == "__main__":
    classifier = VeterinaryNERClassifier()
    
    test_cases = [
        # --- NIVEL 1: PRUEBAS BÁSICAS (Las que ya tenías) ---
        {"desc": "Concepto Clínico", "query": "busco productos clinicos", "expect_types": ["CATEGORIA"]},
        {"desc": "Concepto Comprimido", "query": "necesito comprimido", "expect_types": ["CONCEPTO"]},
        {"desc": "Especie Plural (Perros)", "query": "antibiotico para perros", "expect_types": ["ACCION", "ESPECIE"]},
        {"desc": "Especie Singular (Gato)", "query": "meloxicam gato", "expect_types": ["DROGA", "ESPECIE"]},
        
        # --- NIVEL 2: FUZZY MATCHING (Errores de dedo) ---
        {
            "desc": "Typo en Acción: 'atibiotico' (falta la n)", 
            "query": "atibiotico oral", 
            "expect_types": ["ACCION"] # Debería matchear 'ANTIBIOTICO' por similitud
        },
        {
            "desc": "Typo en Marca: 'roial' por 'royal'", 
            "query": "alimento holliday", 
            "expect_types": ["CATEGORIA", "LABORATORIO"] # 'roial' -> ROYAL, 'alimento' -> CATEGORIA
        },
        {
            "desc": "Typo en Producto: 'sinparica' (n por m)", 
            "query": "precio de sinparica", 
            "expect_types": ["PRODUCTO"] # Debería encontrar 'SIMPARICA'
        },

        # --- NIVEL 3: SINÓNIMOS Y DIMINUTIVOS (Vocabulario Enriquecido) ---
        {
            "desc": "Diminutivo Especie: 'gatitos'", 
            "query": "pipeta para gatitos", 
            "expect_types": ["CONCEPTO", "ESPECIE"] # 'gatitos' debe ser ESPECIE (gracias a _enrich_species)
        },
        {
            "desc": "Sinónimo Especie: 'cachorros'", 
            "query": "vacuna cachorros", 
            "expect_types": ["CONCEPTO", "ESPECIE"] # Asumiendo 'vacuna' en CONCEPTO o CATEGORIA
        },

        # --- NIVEL 4: COMPLEJIDAD Y RANKING ---
        {
            "desc": "Ranking: Droga vs Producto (Bravecto)", 
            "query": "bravecto para perros", 
            "expect_types": ["PRODUCTO", "ESPECIE"] 
            # Aquí 'bravecto' es Producto. Si existe droga 'Fluralaner', no debería confundirse.
            # 'perros' debe ser ESPECIE, no Droga (gracias a tu sanitización).
        },
        {
            "desc": "Multi-Entidad: Lab + Acción + Especie", 
            "query": "antiparasitarios jhon martin para gatos", 
            "expect_types": ["ACCION", "LABORATORIO", "ESPECIE"]
        },
        {
            "desc": "Plurales en Conceptos: 'collares'", 
            "query": "collares para pulgas", 
            "expect_types": ["CONCEPTO", "ESPECIE"] # 'pulgas' suele detectarse como ESPECIE o parte de ACCION
            # Si 'collar' está en conceptos, 'collares' debería salir por Fuzzy o por la 's' extra.
        }
    ]

    print("\n" + "="*80)
    print(f"🧪 EJECUTANDO {len(test_cases)} PRUEBAS (MODO HARDCORE)")
    print("="*80)

    passed = 0
    for i, case in enumerate(test_cases, 1):
        print(f"\n🔹 Caso {i}: {case['desc']}")
        print(f"   Query: '{case['query']}'")
        
        result = classifier.classify(case['query'])
        found_types = [e.entity_type for e in result.all_entities]
        
        # Lógica para verificar que TODOS los tipos esperados estén presentes
        missing = [t for t in case.get('expect_types', []) if t not in found_types]
        
        # Opcional: Verificar que NO haya tipos prohibidos (ej: que 'gatos' no sea DROGA)
        # forbidden = case.get('forbidden_types', [])
        # failed_forbidden = [t for t in forbidden if t in found_types]

        success = not missing # and not failed_forbidden
        
        print(f"   Resultado: {'✅' if success else '❌'}")
        print(f"   → Tipos detectados: {found_types}")
        print(f"   → Entidades: {[e.entity_value for e in result.all_entities]}") # Mostramos qué texto encontró
        
        if missing: print(f"     ⚠️ Faltan: {missing}")
            
        if success: passed += 1

    print("\n" + "="*80)
    print(f"🏁 RESUMEN: {passed}/{len(test_cases)} Pruebas pasadas.")
    print("="*80)