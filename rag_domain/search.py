"""
SEARCH V6.1 - FIXED
- Fix: Solucionado error "semantic_score column does not exist" usando Subquery
- Feature: Boost comercial condicional
- Feature: Top-K adaptativo
"""
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import text
from core.db import get_pgvector_engine
from rag_domain.embeddings import get_embedding_model
from unidecode import unidecode
import re
from difflib import SequenceMatcher


# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════

WEIGHT_SEMANTIC = 1.0
WEIGHT_KEYWORD_FTS = 2.0
WEIGHT_EXACT_MATCH = 3.0

WEIGHT_BRAND_SIMILARITY = 1.0
WEIGHT_NAME_SIMILARITY = 1.5
WEIGHT_WEIGHT_FIT = 1.0
WEIGHT_ATTRIBUTES = 0.5

# Boosts fuertes (cuando el usuario LOS PIDE)
OFFER_BOOST = 2.0
TRANSFER_BOOST = 1.5

# Boosts suaves (cuando NO los pidió pero existen)
OFFER_SOFT_BOOST = 0.5
TRANSFER_SOFT_BOOST = 0.3

THRESHOLD_SPECIFIC = 8.0
THRESHOLD_FAMILY = 5.0
THRESHOLD_CATEGORY = 3.0


class VectorSearchService:
    
    def __init__(self, embedding_service=None):
        self.engine = get_pgvector_engine()
        self.embedding_service = embedding_service or get_embedding_model()

    def _normalize_input(self, text_val: Any) -> str:
        if not text_val:
            return ""
        clean = unidecode(str(text_val)).lower().strip()
        return clean.replace("'", "").replace('"', "").replace(":", "")

    def _parse_weight_range(self, weight_range_str: str) -> Optional[Tuple[float, float]]:
        if not weight_range_str or str(weight_range_str).strip() in ['', '0', 'None', 'null']:
            return None
        
        try:
            parts = str(weight_range_str).split('-')
            if len(parts) == 2:
                min_w = float(parts[0].strip())
                max_w = float(parts[1].strip())
                return (min_w, max_w)
        except (ValueError, AttributeError):
            pass
        
        return None

    def search_with_context(
        self, 
        optimized_data: dict, 
        search_history: List[Dict] = None, 
        top_k: int = 5
    ):
        filters = optimized_data.get('search_filters', {})
        query_text = optimized_data.get('search_input', "")
        
        print(f"\n🔍 [SEARCH V6.1 FIXED] Query: '{query_text}'")
        print(f"🔍 [SEARCH V6.1 FIXED] Filters: {filters}")
        
        # Top-K Adaptativo
        effective_top_k = self._calculate_adaptive_top_k(filters, top_k)
        
        # Preparación
        clean_fts_query = self._normalize_input(query_text)
        enriched_query = self._enrich_query_with_history(query_text, search_history)
        query_vector = self.embedding_service.embed_query(enriched_query)

        # Búsqueda
        raw_results = self._search_with_soft_filters(
            filters, 
            query_vector, 
            clean_fts_query, 
            effective_top_k
        )
        
        print(f"📊 [DB] Candidatos obtenidos: {len(raw_results)}")
        
        # Scoring con boost condicional
        scored = self._apply_gradual_scoring_conditional(raw_results, filters, query_text)
        
        print(f"📊 [SCORING] Candidatos scored: {len(scored)}")
        
        # Threshold
        filtered = self._apply_adaptive_threshold(scored, filters)
        
        print(f"📊 [THRESHOLD] Candidatos filtrados: {len(filtered)}")
        
        # Diversidad (si no pidió ofertas)
        if not filters.get('is_offer') and not filters.get('is_transfer'):
            filtered = self._diversify_results(filtered, effective_top_k)
        
        # Ranking final
        filtered.sort(key=lambda x: x['total_score'], reverse=True)
        
        self._log_results(filtered[:effective_top_k])
        
        return filtered[:effective_top_k]

    def _calculate_adaptive_top_k(self, filters: Dict, base_top_k: int) -> int:
        """Top-k adaptativo según filtros."""
        
        has_specific_filters = any([
            filters.get('weight_min'),
            filters.get('weight_max'),
            filters.get('brand'),
            filters.get('drug'),
            filters.get('exclude_brands'),
            filters.get('presentation'),
            filters.get('species')
        ])
        
        if has_specific_filters:
            effective = min(base_top_k * 2, 15)
            if effective != base_top_k:
                print(f"🎯 [ADAPTIVE] Filtros específicos → top_k: {base_top_k} → {effective}")
            return effective
        
        return base_top_k

    def _search_with_soft_filters(self, filters, query_vector, clean_fts_query, top_k):
        """Query SQL con filtros soft."""
        
        # Primera parte: SELECT con scores calculados
        select_clause = f"""
            SELECT 
                id, entity_type, entity_id, content_text, metadata,
                (1 - (embedding <=> :vector)) as semantic_score,
                CASE 
                    WHEN to_tsvector('spanish', 
                        coalesce(metadata->>'title', '') || ' ' || 
                        coalesce(metadata->>'enterprise_title', '') || ' ' || 
                        coalesce(metadata->>'drug', '')
                    ) @@ websearch_to_tsquery('spanish', :fts_query) 
                    THEN {WEIGHT_KEYWORD_FTS}
                    ELSE 0.0 
                END as keyword_score,
                (
                    CASE WHEN :has_category = true 
                              AND lower(metadata->>'category') = lower(:category)
                         THEN {WEIGHT_EXACT_MATCH}
                         ELSE 0.0 END +
                    CASE WHEN :has_drug = true 
                              AND lower(metadata->>'drug') ILIKE '%' || lower(:drug) || '%'
                         THEN {WEIGHT_EXACT_MATCH} * 0.7
                         ELSE 0.0 END
                )::float as exact_match_score
            FROM embeddings
        """
        
        # Segunda parte: WHERE clauses
        where_clauses = ["1=1"]
        
        params = {
            "vector": str(query_vector),
            "limit": top_k * 20,
            "fts_query": clean_fts_query,
            "has_category": bool(filters.get('category')),
            "category": filters.get('category', ''),
            "has_drug": bool(filters.get('drug')),
            "drug": filters.get('drug', '')
        }

        # Soft filters (construir WHERE dinámicamente)
        if filters.get('brand'):
            brand_norm = self._normalize_input(filters['brand'])
            where_clauses.append("(lower(metadata->>'enterprise_title') ILIKE :brand OR lower(metadata->>'filter_lab') ILIKE :brand)")
            params['brand'] = f'%{brand_norm}%'
        
        if filters.get('categorias'):
            cats = [self._normalize_input(c) for c in filters['categorias']]
            where_clauses.append("lower(metadata->>'category') = ANY(:cats)")
            params['cats'] = cats
        
        if filters.get('species'):
            species_val = filters['species'] if isinstance(filters['species'], list) else [filters['species']]
            species_patterns = [f"%{self._normalize_input(s)}%" for s in species_val]
            where_clauses.append("lower(metadata->>'species_filter') ILIKE ANY(:species_patterns)")
            params['species_patterns'] = species_patterns
        
        if filters.get('presentation'):
            pres = self._normalize_input(filters['presentation'])
            where_clauses.append("lower(metadata->>'presentation') ILIKE :presentation")
            params['presentation'] = f'%{pres}%'
        
        if filters.get('exclude_brands'):
            excluded = [self._normalize_input(b) for b in filters['exclude_brands']]
            where_clauses.append("NOT (lower(metadata->>'enterprise_title') = ANY(:excluded_brands) OR lower(metadata->>'filter_lab') = ANY(:excluded_brands))")
            params['excluded_brands'] = excluded
        
        # Ofertas/Transfers
        offer_clauses = []
        if filters.get('is_offer'):
            offer_clauses.append("(metadata->>'is_offer')::boolean = true")
            params['has_offer'] = True
        
        if filters.get('is_transfer'):
            offer_clauses.append("(metadata->>'has_transfer')::boolean = true")
            params['has_transfer'] = True
        
        if len(offer_clauses) > 1:
            where_clauses.append(f"({' OR '.join(offer_clauses)})")
        elif len(offer_clauses) == 1:
            where_clauses.append(offer_clauses[0])
        
        # Construir query final
        where_clause = " AND ".join(where_clauses)
        
        # ═══════════════════════════════════════════════════════════
        # FIX: WRAP EN SUBQUERY
        # Postgres no permite usar alias (semantic_score) en formulas del ORDER BY
        # sin un wrapper o CTE.
        # ═══════════════════════════════════════════════════════════
        final_sql = f"""
            SELECT * FROM (
                {select_clause}
                WHERE {where_clause}
            ) as candidates
            ORDER BY (
                semantic_score * {WEIGHT_SEMANTIC} + 
                keyword_score + 
                exact_match_score
            ) DESC
            LIMIT :limit
        """

        with self.engine.connect() as conn:
            return conn.execute(text(final_sql), params).fetchall()

    def _apply_gradual_scoring_conditional(self, raw_results, filters, query_text):
        """Scoring con boost CONDICIONAL."""
        
        scored_results = []
        user_requested_offers = filters.get('is_offer', False)
        user_requested_transfers = filters.get('is_transfer', False)
        
        for row in raw_results:
            semantic_score = float(row.semantic_score) if hasattr(row, 'semantic_score') else 0.0
            keyword_score = float(row.keyword_score) if hasattr(row, 'keyword_score') else 0.0
            exact_match_score = float(row.exact_match_score) if hasattr(row, 'exact_match_score') else 0.0
            
            brand_score = 0.0
            if filters.get('brand'):
                brand_score = self._score_brand_similarity(
                    str(row.metadata.get('enterprise_title', '')), 
                    filters['brand']
                )
            
            name_score = 0.0
            if query_text:
                name_score = self._score_name_similarity(
                    str(row.metadata.get('title', '')),
                    query_text
                )
            
            weight_score = 0.0
            if filters.get('weight_min') or filters.get('weight_max'):
                weight_range_str = row.metadata.get('weight_range')
                weight_range = self._parse_weight_range(weight_range_str)
                if weight_range:
                    weight_score = self._score_weight_fit(
                        weight_range,
                        filters.get('weight_min', 0),
                        filters.get('weight_max', 999)
                    )
            
            attribute_score = self._score_attributes(row.metadata, filters)
            
            # ═══ BOOST CONDICIONAL (NUEVO) ═══
            commercial_boost = 0.0
            product_is_offer = self._is_true(row.metadata.get('is_offer'))
            product_has_transfer = self._is_true(row.metadata.get('has_transfer'))
            
            if product_is_offer:
                commercial_boost += OFFER_BOOST if user_requested_offers else OFFER_SOFT_BOOST
            
            if product_has_transfer:
                commercial_boost += TRANSFER_BOOST if user_requested_transfers else TRANSFER_SOFT_BOOST
            
            total_score = (
                semantic_score * WEIGHT_SEMANTIC +
                keyword_score +
                exact_match_score +
                brand_score * WEIGHT_BRAND_SIMILARITY +
                name_score * WEIGHT_NAME_SIMILARITY +
                weight_score * WEIGHT_WEIGHT_FIT +
                attribute_score * WEIGHT_ATTRIBUTES +
                commercial_boost
            )
            
            scored_results.append({
                'entity_id': row.entity_id,
                'entity_type': row.entity_type,
                'content': row.content_text,
                'metadata': row.metadata,
                'total_score': total_score,
                '_debug': {
                    'semantic': semantic_score,
                    'keyword': keyword_score,
                    'exact': exact_match_score,
                    'brand_sim': brand_score,
                    'name_sim': name_score,
                    'weight_fit': weight_score,
                    'attributes': attribute_score,
                    'commercial': commercial_boost,
                    'is_offer': product_is_offer,
                    'has_transfer': product_has_transfer
                }
            })
        
        return scored_results

    def _diversify_results(self, results, top_k):
        """Asegura diversidad 70/30."""
        if not results:
            return results
        
        offers = []
        products = []
        
        for r in results:
            is_offer = self._is_true(r['metadata'].get('is_offer'))
            has_transfer = self._is_true(r['metadata'].get('has_transfer'))
            
            if is_offer or has_transfer:
                offers.append(r)
            else:
                products.append(r)
        
        if len(products) < 2 or len(offers) < 1:
            return results[:top_k]
        
        target_products = int(top_k * 0.7)
        target_offers = top_k - target_products
        
        diversified = []
        diversified.extend(products[:target_products])
        diversified.extend(offers[:target_offers])
        
        diversified.sort(key=lambda x: x['total_score'], reverse=True)
        
        print(f"🎨 [DIVERSITY] {len(products[:target_products])} productos + {len(offers[:target_offers])} ofertas")
        
        return diversified

    def _score_brand_similarity(self, product_brand, filter_brand):
        product_clean = product_brand.lower().strip()
        filter_clean = filter_brand.lower().strip()
        
        if not product_clean or not filter_clean:
            return 0.0
        
        if product_clean == filter_clean:
            return 5.0
        
        if filter_clean in product_clean or product_clean in filter_clean:
            return 3.0
        
        similarity = SequenceMatcher(None, product_clean, filter_clean).ratio()
        
        return 2.0 if similarity > 0.8 else (1.0 if similarity > 0.6 else 0.0)

    def _score_name_similarity(self, product_name, query_name):
        product_tokens = set(self._normalize_input(product_name).split())
        query_tokens = set(self._normalize_input(query_name).split())
        
        stop_words = {'de', 'para', 'con', 'sin', 'x', 'y', 'en'}
        product_tokens -= stop_words
        query_tokens -= stop_words
        
        if not query_tokens:
            return 0.0
        
        common = product_tokens & query_tokens
        union = product_tokens | query_tokens
        
        jaccard = len(common) / len(union) if union else 0.0
        
        if jaccard >= 0.8:
            return 5.0
        elif jaccard >= 0.6:
            return 4.0
        elif jaccard >= 0.4:
            return 2.0
        elif jaccard >= 0.2:
            return 1.0
        else:
            return 0.0

    def _score_weight_fit(self, product_weight_range, target_min, target_max):
        if not product_weight_range:
            return 0.0
        
        product_min, product_max = product_weight_range
        
        if product_min >= target_min and product_max <= target_max:
            return 4.0
        
        if not (product_max < target_min or product_min > target_max):
            overlap_min = max(product_min, target_min)
            overlap_max = min(product_max, target_max)
            overlap_size = overlap_max - overlap_min
            target_size = target_max - target_min
            
            overlap_ratio = overlap_size / target_size if target_size > 0 else 0
            
            if overlap_ratio > 0.7:
                return 3.5
            elif overlap_ratio > 0.5:
                return 3.0
            elif overlap_ratio > 0.3:
                return 2.5
            else:
                return 2.0
        
        if product_max < target_min:
            distance = target_min - product_max
        else:
            distance = product_min - target_max
        
        target_range = target_max - target_min
        relative_distance = distance / target_range if target_range > 0 else 999
        
        if relative_distance < 0.2:
            return 2.0
        elif relative_distance < 0.5:
            return 1.0
        else:
            return 0.0

    def _score_attributes(self, metadata, filters):
        score = 0.0
        
        if filters.get('category'):
            if filters['category'].lower() in str(metadata.get('category', '')).lower():
                score += 0.5
        
        if filters.get('presentation'):
            if filters['presentation'].lower() in str(metadata.get('presentation', '')).lower():
                score += 0.5
        
        if filters.get('species'):
            filter_species = filters['species'].lower() if isinstance(filters['species'], str) else ''
            if filter_species and filter_species in str(metadata.get('species_filter', '')).lower():
                score += 0.5
        
        if filters.get('drug'):
            if filters['drug'].lower() in str(metadata.get('drug', '')).lower():
                score += 0.5
        
        return score

    def _apply_adaptive_threshold(self, scored, filters):
        if filters.get('target_products'):
            threshold = THRESHOLD_SPECIFIC
            search_type = "ESPECÍFICA"
        elif filters.get('brand') or filters.get('laboratorios'):
            threshold = THRESHOLD_FAMILY
            search_type = "FAMILIA"
        else:
            threshold = THRESHOLD_CATEGORY
            search_type = "CATEGORÍA"
        
        print(f"🎯 [THRESHOLD] Tipo: {search_type} | Mínimo: {threshold:.1f}")
        
        filtered = [c for c in scored if c['total_score'] >= threshold]
        
        if len(filtered) < 3 and len(scored) >= 3:
            print(f"⚠️ [THRESHOLD] Solo {len(filtered)} sobre threshold, retornando top 3")
            filtered = scored[:3]
        
        if len(scored) - len(filtered) > 0:
            print(f"   ❌ Descartados {len(scored) - len(filtered)} por score bajo")
        
        return filtered

    def _build_offer_filter(self, filters):
        offer_flags = []
        
        if filters.get('is_offer'):
            offer_flags.append("(metadata->>'is_offer')::boolean = true")
        
        if filters.get('is_transfer'):
            offer_flags.append("(metadata->>'has_transfer')::boolean = true")
        
        if len(offer_flags) > 1:
            return f" AND ({' OR '.join(offer_flags)})"
        elif len(offer_flags) == 1:
            return f" AND {offer_flags[0]}"
        else:
            return ""

    def _is_true(self, value):
        if isinstance(value, bool):
            return value
        return str(value).lower() in ('true', '1', 'yes', 'si')
    
    def _enrich_query_with_history(self, current_query, search_history):
        if not search_history or not current_query:
            return current_query
        
        recent = search_history[-2:]
        history_entities = []
        
        for item in recent:
            all_entities = item.get('all_entities', [])
            sorted_entities = sorted(all_entities, key=lambda e: e.get('score', 0), reverse=True)
            
            for ent in sorted_entities[:2]:
                if isinstance(ent, dict):
                    value = ent.get('value', '')
                    if value and value not in current_query:
                        history_entities.append(value)
        
        if history_entities:
            unique_entities = []
            seen = set()
            for ent in history_entities[:3]:
                if ent.lower() not in seen:
                    seen.add(ent.lower())
                    unique_entities.append(ent)
            
            return f"{current_query} {' '.join(unique_entities)}"
        
        return current_query

    def _log_results(self, results):
        print(f"\n📊 [RESULTS] Top {len(results)}:\n")
        
        for idx, r in enumerate(results, 1):
            meta = r.get('metadata', {})
            debug = r.get('_debug', {})
            
            title = meta.get('title', 'N/A')[:50]
            lab = meta.get('enterprise_title', 'N/A')
            
            indicators = []
            if debug.get('is_offer'):
                indicators.append("🏷️")
            if debug.get('has_transfer'):
                indicators.append("🎁")
            
            indicator_str = "".join(indicators) if indicators else ""
            
            print(f"   {idx}. {title} {indicator_str}")
            print(f"      Lab: {lab}")
            print(f"      Score: {r['total_score']:.2f} | "
                  f"sem:{debug.get('semantic', 0):.2f} "
                  f"kw:{debug.get('keyword', 0):.1f} "
                  f"brand:{debug.get('brand_sim', 0):.1f} "
                  f"name:{debug.get('name_sim', 0):.1f} "
                  f"weight:{debug.get('weight_fit', 0):.1f} "
                  f"attr:{debug.get('attributes', 0):.1f} "
                  f"comm:{debug.get('commercial', 0):.1f}")
            print()