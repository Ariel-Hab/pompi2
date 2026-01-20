"""
SEARCH V3.0 - Identity-focused with Fuzzy NER Matching
Separación clara: Vector = Identidad | Metadata = Contexto LLM
"""
from typing import List, Dict, Any
from sqlalchemy import text
from core.db import get_pgvector_engine
from rag_domain.embeddings import get_embedding_model
from unidecode import unidecode

# Configuración de Pesos (REBALANCEADOS)
WEIGHT_SEMANTIC = 1.0
WEIGHT_KEYWORD_FTS = 2.0
WEIGHT_NER_SIMILARITY = 4.0  # Fuzzy match con pg_trgm
WEIGHT_DOSAGE = 0.5

class VectorSearchService:
    
    def __init__(self, embedding_service=None):
        self.engine = get_pgvector_engine()
        self.embedding_service = embedding_service or get_embedding_model()

    def _normalize_input(self, text_val: Any) -> str:
        """Normalización estándar para búsqueda FTS (mantiene espacios)"""
        if not text_val: return ""
        clean = unidecode(str(text_val)).lower().strip()
        return clean.replace("'", "").replace('"', "").replace(":", "")

    def _normalize_strict(self, text_val: Any) -> str:
        """
        Normalización agresiva para similarity matching.
        Elimina espacios, puntos y guiones para comparar "40mg" con "40 mg".
        """
        if not text_val: return ""
        import re
        from unidecode import unidecode
        text_clean = unidecode(str(text_val)).lower()
        return re.sub(r'[^a-z0-9]', '', text_clean)

    def search_with_context(
        self, 
        optimized_data: dict, 
        search_history: List[Dict] = None, 
        top_k: int = 5
    ):
        # 1. Preparación de Datos
        filters = optimized_data.get('search_filters', {})
        query_text = optimized_data.get('search_input', "")
        
        # ✅ CAMBIO: Ya NO usamos main_entity, solo el query_text limpio
        clean_fts_query = self._normalize_input(query_text)
        
        # 2. Vector Query
        enriched_query = self._enrich_query_with_history(query_text, search_history)
        query_vector = self.embedding_service.embed_query(enriched_query)

        # 3. Preparación para Fuzzy NER Matching
        raw_targets = filters.get('target_products', [])
        normalized_targets = [self._normalize_strict(t) for t in raw_targets]
        has_target_products = len(normalized_targets) > 0
        
        # 4. Construcción SQL Híbrida
        # FIX: Added ::text[] cast to unnest(:normalized_targets) to prevent ambiguous function error
        base_sql = f"""
            SELECT 
                id, 
                entity_type, 
                entity_id, 
                content_text, 
                metadata,
                
                -- A. Score Semántico
                (1 - (embedding <=> :vector)) as semantic_score,
                
                -- B. Score Full Text Search (DICCIONARIO SPANISH)
                CASE 
                    WHEN to_tsvector('spanish', 
                        coalesce(metadata->>'title', '') || ' ' || 
                        coalesce(metadata->>'enterprise_title', '') || ' ' || 
                        coalesce(metadata->>'search_keywords', '')
                    ) @@ websearch_to_tsquery('spanish', :fts_query) 
                    THEN {WEIGHT_KEYWORD_FTS}
                    ELSE 0.0 
                END as keyword_score,

                -- C. Score de Dosis
                CASE 
                    WHEN :target_dosage > 0 
                         AND (metadata->>'dosage_value') IS NOT NULL 
                         AND (metadata->>'dosage_value')::text ~ '^[0-9.]+$'
                    THEN 
                        LEAST(
                            {WEIGHT_DOSAGE}::float,
                            {WEIGHT_DOSAGE}::float / (1 + ABS(
                                (metadata->>'dosage_value')::float - :target_dosage
                            ))
                        )
                    ELSE 0.0::float
                END as proximity_score,

                -- D. Score NER Fuzzy Similarity (pg_trgm)
                CASE 
                    WHEN :has_target_products = true 
                    THEN (
                        SELECT MAX(similarity(
                            translate(lower(unaccent(metadata->>'title')), ' .-/', ''), 
                            t
                        ))
                        FROM unnest(:normalized_targets :: text[]) AS t
                    ) * {WEIGHT_NER_SIMILARITY}
                    ELSE 0.0
                END as ner_similarity_score

            FROM embeddings
            WHERE 1=1
        """
        
        params = {
            "vector": str(query_vector), 
            "limit": top_k * 10,
            "target_dosage": filters.get('dosage_value', 0),
            "fts_query": clean_fts_query,
            "has_target_products": has_target_products,
            "normalized_targets": normalized_targets
        }

        # 5. FILTROS DUROS (HARD FILTERS)
        
        # A. Laboratorio
        if filters.get('laboratorios'):
            labs = [self._normalize_input(l) for l in filters['laboratorios']]
            base_sql += " AND (metadata->>'filter_lab' = ANY(:labs))"
            params['labs'] = labs

        # B. Categoría
        if filters.get('categorias'):
            cats = [self._normalize_input(c) for c in filters['categorias']]
            base_sql += " AND (metadata->>'filter_category' = ANY(:cats))"
            params['cats'] = cats

        # C. Especie Smart (singular + plural)
        if filters.get('species'):
            species_patterns = []
            for s in filters['species']:
                base = self._normalize_input(s)
                species_patterns.append(f"%{base}%")
                if base.endswith('s'):
                    species_patterns.append(f"%{base[:-1]}%")
            
            base_sql += " AND (metadata->>'species_filter' ILIKE ANY(:species_patterns))"
            params['species_patterns'] = species_patterns

        # D. Ofertas
        if filters.get('is_offer'):
            base_sql += " AND (metadata->>'is_offer')::boolean = true"

        # 6. Ordenamiento (Incluye ner_similarity_score)
        # FIX: Added ::text[] cast here as well
        base_sql += """
            ORDER BY (
                ((1 - (embedding <=> :vector)) * :w_semantic) + 
                (CASE 
                    WHEN to_tsvector('spanish', 
                        coalesce(metadata->>'title', '') || ' ' || 
                        coalesce(metadata->>'enterprise_title', '') || ' ' || 
                        coalesce(metadata->>'search_keywords', '')
                    ) @@ websearch_to_tsquery('spanish', :fts_query) 
                    THEN :w_keyword
                    ELSE 0.0 
                 END) +
                (CASE 
                    WHEN :has_target_products = true 
                    THEN (
                        SELECT MAX(similarity(
                            translate(lower(unaccent(metadata->>'title')), ' .-/', ''), 
                            t
                        ))
                        FROM unnest(:normalized_targets :: text[]) AS t
                    ) * :w_ner
                    ELSE 0.0
                END)
            ) DESC
            LIMIT :limit
        """
        
        params['w_semantic'] = WEIGHT_SEMANTIC
        params['w_keyword'] = WEIGHT_KEYWORD_FTS
        params['w_ner'] = WEIGHT_NER_SIMILARITY

        # 7. Ejecución
        with self.engine.connect() as conn:
            raw_results = conn.execute(text(base_sql), params).fetchall()

        # 8. Post-Procesamiento
        candidates = []
        for row in raw_results:
            total_score = (
                float(row.semantic_score) * WEIGHT_SEMANTIC +
                float(row.keyword_score) + 
                float(row.proximity_score) +
                float(row.ner_similarity_score)
            )
            
            candidates.append({
                'entity_id': row.entity_id,
                'entity_type': row.entity_type,
                'content': row.content_text,
                'metadata': row.metadata,
                'total_score': total_score,
                
                # Scores individuales
                'semantic_score': float(row.semantic_score),
                'keyword_score': float(row.keyword_score),
                'proximity_score': float(row.proximity_score),
                'ner_similarity_score': float(row.ner_similarity_score),
                'match_boost': float(row.ner_similarity_score),
                
                # Debug
                '_debug': {
                    'sem': float(row.semantic_score), 
                    'key': float(row.keyword_score),
                    'ner_sim': float(row.ner_similarity_score),
                    'prox': float(row.proximity_score)
                }
            })
            
        candidates.sort(key=lambda x: x['total_score'], reverse=True)
        return self._format_results(candidates, top_k)

    def _format_results(self, results: List[Dict], top_k: int) -> List[Dict]:
        return results[:top_k]
    
    def _enrich_query_with_history(self, current_query: str, search_history: List[Dict]) -> str:
        if not search_history: return current_query
        return current_query