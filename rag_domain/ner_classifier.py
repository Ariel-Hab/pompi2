"""
NER Classifier v3.2 - FINAL
- Hybrid: Token Scoring + Root Fuzzy Fallback
- Keywords expandidos para ofertas/transfers
- Detección mejorada de promociones
"""
import re
import csv
import difflib
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class EntityMatch:
    entity_type: str 
    entity_value: str
    position: int
    length: int
    original_text: str = ""
    score: float = 1.0

    @property
    def match_diff(self) -> int:
        """Penalización por diferencia de longitud (Ruido vs Señal)."""
        if not self.original_text: 
            return 0
        return abs(len(self.original_text) - self.length)

@dataclass
class ClassificationResult:
    entity_type: Optional[str]
    entity_value: Optional[str]
    all_entities: List[EntityMatch] = field(default_factory=list)
    filters: Dict = field(default_factory=dict)
    intent: str = "SEARCH"
    confidence: float = 0.5


class VeterinaryNERClassifier:
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            self.data_path = project_root / "data"
        else:
            self.data_path = Path(data_dir)

        # Palabras funcionales a ignorar
        self.ignored_terms = {
            'para', 'con', 'sin', 'los', 'las', 'una', 'unos', 'del', 'por', 'de',
            'uso', 'veterinario', 'envase', 'caja', 'peso', 'vivo', 'kpv', 'kilos',
            'comprimido', 'comprimidos', 'ml', 'mg', 'gr', 'accion', 'terapeutica',
            'tratamiento', 'enfermedades', 'amplio', 'espectro', 'via', 'oral',
            'producto', 'productos', 'lista', 'precio', 'venta', 'h/', 'x', 'y'
        }
        
        self.stop_phrases = [
            "que productos", "quien vende", "quien tiene", "que tiene", "busco", 
            "necesito", "precio de", "tenes", "tienen", "tiene",
            "info de", "productos de", "informacion de", 
            "dame", "quiero", "me das", "consulta sobre", "decime", "contame", "mostrame"
        ]
        
        self.smalltalk_keywords = {'hola', 'hey', 'buenos', 'dias', 'tardes', 'como', 'andas', 'gracias', 'chau'}
        self.recommendation_keywords = {'recomiend', 'sugier', 'que usar', 'ayuda con', 'sirve para'}
        
        # ═══════════════════════════════════════════════════════════
        # KEYWORDS EXPANDIDOS - Ofertas/Transfers
        # ═══════════════════════════════════════════════════════════
        self.offer_keywords = {
            # Español - términos comunes
            'oferta', 'ofertas', 'promo', 'promos', 'promocion', 'promociones',
            'descuento', 'descuentos', 'rebaja', 'rebajas',
            'liquidacion', 'especial', 'especiales',
            'barato', 'baratos', 'oferton',
            # Abreviaturas
            'off', 'desc', 'dscto', 'dcto', 'promoc',
            # Combinaciones numéricas
            '2x1', '3x2', '4x3'
        }
        
        self.transfer_keywords = {
            'transfer', 'transfers', 
            'bonif', 'bonificacion', 'bonificaciones',
            'regalo', 'regalos', 'regla', 'reglas',
            'combo', 'combos', 'pack', 'packs',
            'promo pack', 'bm'
        }
        
        # --- CARGA DE DATOS ---
        vademecum_path = self.data_path / "vademecum.csv"
        enterprises_path = self.data_path / "enterprises.csv"

        self.laboratorios = self._load_known_entities(enterprises_path, "title", tokenize=True)
        self.productos = self._load_known_entities(vademecum_path, "PRODUCTO")
        self.drogas = self._load_known_entities(vademecum_path, "PRINCIPIO ACTIVO", split_chars=[',', '+', '/'], tokenize=True)
        self.categorias = self._load_known_entities(vademecum_path, "CATEGORIA", tokenize=True)
        self.acciones = self._load_known_entities(vademecum_path, "ACCION TERAPEUTICA", split_chars=[',', '-', '/', '.', ':'], tokenize=True)
        self.conceptos = self._load_known_entities(vademecum_path, "CONCEPTO", split_chars=['/'], tokenize=False)
        self.especies = self._load_known_entities(vademecum_path, "ESPECIE", split_chars=[',', ';', '/'], tokenize=True)
        
        self._enrich_species_vocabulary()
        self._sanitize_entities() 
        
        # Pre-computar tokens para scoring rápido
        self.product_tokens_map = self._precompute_product_tokens()
        # Pre-computar raíces para fuzzy rápido
        self.product_roots = self._precompute_product_roots()

    def _precompute_product_tokens(self) -> List[Tuple[str, Set[str]]]:
        """Pre-calcula tokens de productos para ranking rápido."""
        processed = []
        for prod in self.productos:
            tokens = set(self._tokenize_semantic(prod))
            if tokens:
                processed.append((prod, tokens))
        return processed

    def _precompute_product_roots(self) -> Dict[str, List[str]]:
        """
        Mapea 'Raíz' (primer token) -> Lista de Productos.
        Usado para corregir typos como 'Sinparica' -> 'Simparica ...'
        """
        roots = {}
        for prod in self.productos:
            tokens = self._tokenize_semantic(prod)
            if tokens:
                root = tokens[0].upper()
                if root not in roots: roots[root] = []
                roots[root].append(prod)
        return roots

    def _tokenize_semantic(self, text: str) -> List[str]:
        """Tokeniza y limpia texto para comparación semántica."""
        clean = self._normalize_text(text)
        clean = re.sub(r'(\d+)([a-z]+)', r'\1 \2', clean) 
        tokens = clean.split()
        return [t for t in tokens if t not in self.ignored_terms and len(t) > 1]

    def _enrich_species_vocabulary(self):
        synonyms = {
            'canino', 'caninos', 'felino', 'felinos', 'cachorro', 'cachorros', 
            'gatito', 'gatitos', 'equino', 'equinos', 'bovino', 'bovinos', 
            'perro', 'perros', 'gato', 'gatos', 'pulga', 'pulgas', 'garrapata'
        }
        self.especies.update({s.upper() for s in synonyms})
        extras = set()
        for sp in self.especies:
            sp_lower = sp.lower()
            if sp_lower.endswith(('a', 'e', 'i', 'o', 'u')): extras.add(sp + "S")
            elif sp_lower[-1].isalpha(): extras.add(sp + "ES")
        self.especies.update(extras)

    def _sanitize_entities(self):
        stop_words_veterinaria = {'PARA', 'USO', 'VETERINARIO', 'DEL', 'LOS', 'LAS', 'CON', 'SIN'}
        measurement_pattern = re.compile(r'^\d+\s*(KG|G|GR|MG|ML|L|CC|CM)$', re.IGNORECASE)

        def is_valid(text):
            if text in stop_words_veterinaria: return False
            if text in self.especies: return False
            if len(text) < 2: return False
            if measurement_pattern.match(text): return False 
            return True
        
        self.drogas = {d for d in self.drogas if is_valid(d)}
        self.acciones = {a for a in self.acciones if is_valid(a)}
        manual_concepts = {'COMPRIMIDO', 'COMPRIMIDOS', 'JARABE', 'GOTAS', 'PIPETA', 'COLLAR', 'COLLARES', 'VACUNA', 'VACUNAS'}
        self.conceptos.update(manual_concepts)
        self.categorias.update({'CLINICO', 'FARMACIA', 'ALIMENTO', 'ACCESORIO'})

    def classify(self, query: str) -> ClassificationResult:
        query_norm = self._normalize_text(query)
        query_clean = self._clean_query(query_norm)
        intent = self._detect_intent(query_norm, query_clean)
        
        if intent == "SMALLTALK" and not self._mentions_products(query_clean):
            return ClassificationResult(None, None, [], {}, "SMALLTALK", 1.0)
        
        all_entities = self._find_all_entities(query_clean, query_norm)
        filters = self._extract_filters(query_norm)
        confidence = self._calculate_confidence(all_entities, filters)
        
        return ClassificationResult(
            entity_type=None,
            entity_value=None,
            all_entities=all_entities,
            filters=filters,
            intent=intent,
            confidence=confidence
        )
    
    def _find_all_entities(self, query_clean: str, query_norm: str) -> List[EntityMatch]:
        matches = []
        query_lower = query_clean.lower()
        query_words = query_clean.split()
        
        # 1. Búsqueda Exacta (Para entidades no-producto)
        def search_exact(entity_set, type_name):
            for item in entity_set:
                item_lower = item.lower()
                
                # Usar word boundaries
                pattern = r'(?<!\w)' + re.escape(item_lower) + r'(?!\w)'
                match = re.search(pattern, query_lower)
                if match:
                    pos = match.start()
                    if not self._is_hard_overlapping(pos, len(item_lower), matches):
                        matches.append(EntityMatch(type_name, item, pos, len(item_lower), original_text=item, score=1.0))

        search_exact(self.laboratorios, "LABORATORIO")
        search_exact(self.drogas, "DROGA")
        search_exact(self.acciones, "ACCION")
        search_exact(self.categorias, "CATEGORIA")
        search_exact(self.conceptos, "CONCEPTO")
        search_exact(self.especies, "ESPECIE")

        # 2. PRODUCTOS: SCORING + FUZZY ROOT
        query_tokens = set(self._tokenize_semantic(query_clean))
        matched_products_indices = set()

        if query_tokens:
            product_matches = []
            for prod_name, prod_tokens in self.product_tokens_map:
                intersection = query_tokens.intersection(prod_tokens)
                if not intersection: continue
                
                query_coverage = len(intersection) / len(query_tokens)
                if query_coverage > 0.3:
                    try:
                        first_match_token = list(intersection)[0]
                        idx = query_lower.find(first_match_token)
                        if idx == -1: idx = 0
                    except: idx = 0
                    
                    product_matches.append(EntityMatch(
                        "PRODUCTO", prod_name, idx, len(prod_name),
                        original_text=prod_name, score=query_coverage
                    ))
                    matched_products_indices.add(prod_name)

            product_matches.sort(key=lambda x: x.score, reverse=True)
            matches.extend(product_matches[:40])

        # Fuzzy Root para typos
        for word in query_words:
            if len(word) < 4 or word.lower() in self.ignored_terms: continue
            
            candidates = difflib.get_close_matches(word.upper(), self.product_roots.keys(), n=1, cutoff=0.85)
            
            if candidates:
                root_matched = candidates[0]
                possible_products = self.product_roots[root_matched]
                word_pos = query_lower.find(word.lower())
                
                for prod in possible_products:
                    if prod in matched_products_indices: continue
                    
                    matches.append(EntityMatch(
                        "PRODUCTO", prod, word_pos, len(prod),
                        original_text=prod, score=0.85
                    ))
                    matched_products_indices.add(prod)

        # 3. Fuzzy Matching (Resto de entidades)
        fuzzy_targets = {
            "CATEGORIA": self.categorias, "ESPECIE": self.especies,
            "CONCEPTO": self.conceptos, "ACCION": self.acciones,
            "LABORATORIO": self.laboratorios, "DROGA": self.drogas 
        }

        for word in query_words:
            if len(word) < 4 or word.lower() in self.ignored_terms: continue
            word_pos = query_lower.find(word.lower())
            
            if word_pos != -1 and self._is_hard_overlapping(word_pos, len(word), matches): continue

            for type_name, entity_set in fuzzy_targets.items():
                candidates = difflib.get_close_matches(word.upper(), entity_set, n=1, cutoff=0.85)
                if candidates:
                    match_val = candidates[0]
                    matches.append(EntityMatch(
                        type_name, match_val, word_pos, len(word), 
                        original_text=match_val, score=0.85
                    ))
                    break

        # RANKING FINAL
        matches.sort(key=lambda m: (
            -m.score,
            self._get_entity_type_priority(m.entity_type),
            m.position
        ))
        
        # DEDUPLICACIÓN
        final_matches = []
        seen_entities = set()
        for m in matches:
            key = (m.entity_type, m.entity_value)
            if key not in seen_entities:
                seen_entities.add(key)
                final_matches.append(m)

        return final_matches
    
    def _is_hard_overlapping(self, position, length, existing_matches):
        new_end = position + length
        for match in existing_matches:
            if match.position == position and match.length == length: continue 
            existing_end = match.position + match.length
            if not (new_end <= match.position or position >= existing_end):
                if match.score > 0.9: return True
        return False
    
    def _get_entity_type_priority(self, entity_type: str) -> int:
        priorities = {
            "PRODUCTO": 1, "LABORATORIO": 2, "CATEGORIA": 3,
            "DROGA": 4, "ACCION": 5, "CONCEPTO": 6, "ESPECIE": 7
        }
        return priorities.get(entity_type, 99)

    def _load_known_entities(self, csv_path: Path, column: str, split_chars: List[str] = None, tokenize: bool = False) -> Set[str]:
        entities = set()
        if not csv_path.exists(): return entities
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_val = row.get(column, '').strip()
                if not raw_val or raw_val in ['0', 'N/A', 'None']: continue
                values_to_process = [raw_val]
                if split_chars:
                    temp = [raw_val]
                    for char in split_chars:
                        new_temp = []
                        for v in temp: new_temp.extend(v.split(char))
                        temp = new_temp
                    values_to_process = [v.strip() for v in temp if v.strip()]
                
                for val in values_to_process:
                    val_clean = re.sub(r'\(.*?\)', '', val).strip().strip(".:;,")
                    if val_clean.lower() in self.ignored_terms: continue
                    if len(val_clean) > 2:
                        entities.add(val_clean.upper()) 
                        if tokenize:
                            words = val_clean.split()
                            if len(words) > 1:
                                for w in words:
                                    w_clean = w.strip(".:;,")
                                    if len(w_clean) > 3 and w_clean.lower() not in self.ignored_terms:
                                        entities.add(w_clean.upper())
        return entities

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
        return re.sub(r'\s+', ' ', text)
    
    def _clean_query(self, text: str) -> str:
        cleaned = text
        for phrase in self.stop_phrases: cleaned = cleaned.replace(phrase, "")
        return re.sub(r'[?¿!¡]', '', cleaned).strip()

    def _detect_intent(self, query_norm: str, query_clean: str) -> str:
        if (any(kw in query_norm for kw in self.smalltalk_keywords) and not self._mentions_products(query_clean)):
            return "SMALLTALK"
        if any(kw in query_norm for kw in self.recommendation_keywords): return "RECOMMENDATION"
        return "SEARCH"

    def _extract_filters(self, query: str) -> dict:
        filters = {}
        
        # Dosage/Peso
        match_dose = re.search(r'(\d+(?:[,\.]\d+)?)\s*(mg|ml|gr?|kg|kilos)', query)
        if match_dose:
            filters['dosage_value'] = float(match_dose.group(1).replace(',', '.'))
            filters['dosage_unit'] = match_dose.group(2)
        
        # Presentación
        pres_map = {'comprimido': 'comprimidos', 'tableta': 'comprimidos', 'inyectable': 'inyectable'}
        for k, v in pres_map.items():
            if k in query: filters['presentation'] = v
        
        # ═══════════════════════════════════════════════════════════
        # DETECCIÓN DE OFERTAS/TRANSFERS - Mejorada
        # ═══════════════════════════════════════════════════════════
        query_lower = query.lower()
        
        # Ofertas
        if any(kw in query_lower for kw in self.offer_keywords): 
            filters['is_offer'] = True
        
        # Transfers
        if any(kw in query_lower for kw in self.transfer_keywords): 
            filters['is_transfer'] = True
        
        return filters

    def _mentions_products(self, query: str) -> bool:
        words = query.lower().split()
        meaningful_words = [w for w in words if w not in self.smalltalk_keywords and w not in self.stop_phrases and len(w) > 2]
        return len(meaningful_words) > 0
    
    def _calculate_confidence(self, entities, filters):
        return min(0.5 + len(entities) * 0.2 + (0.1 if filters else 0), 1.0)


def classification_to_optimizer_format(classification: ClassificationResult) -> Dict:
    """
    Convierte a formato Optimizer usando TODAS las entidades.
    """
    labs = [e.entity_value for e in classification.all_entities if e.entity_type == "LABORATORIO"]
    cats = [e.entity_value for e in classification.all_entities if e.entity_type == "CATEGORIA"]
    actions = [e.entity_value for e in classification.all_entities if e.entity_type == "ACCION"]
    products = [e.entity_value for e in classification.all_entities if e.entity_type == "PRODUCTO"]
    species = [e.entity_value for e in classification.all_entities if e.entity_type == "ESPECIE"]

    context_parts = []
    context_parts.extend(products)
    context_parts.extend(labs)
    context_parts.extend(cats)
    context_parts.extend(actions)
    
    details_text = " ".join(context_parts)
    classification.filters['target_products'] = products
    
    return {
        "intent": classification.intent,
        "search_input": details_text,
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