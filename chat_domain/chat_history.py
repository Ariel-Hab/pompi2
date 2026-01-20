"""
CHAT HISTORY - Multi-Entity (SIN redundancia de entity_searched)
Solo usa all_entities_json para almacenar todas las entidades
"""
from typing import List, Dict, Optional
from sqlalchemy import text
from core.db import get_pgvector_engine
import json


class ChatHistoryService:
    def __init__(self, session_id: str, user_id: str):
        self.engine = get_pgvector_engine()
        self.session_id = session_id
        self.user_id = user_id

    def add_message(
        self, 
        role: str, 
        content: str, 
        rag_context: str = None, 
        classification: str = None,
        all_entities: List[Dict] = None  # ← ÚNICO campo de entidades
    ):
        """
        Guarda un mensaje individual con TODAS las entidades detectadas.
        
        SIMPLIFICADO: Solo usa all_entities_json (sin redundancia)
        
        Args:
            all_entities: Lista de dicts con {type, value, position}
        """
        # Serializar all_entities a JSON
        entities_json = json.dumps(all_entities) if all_entities else None
        
        query = text("""
            INSERT INTO chat_history (
                session_id, user_id, role, content, 
                rag_context, classification, all_entities_json
            )
            VALUES (
                :session_id, :user_id, :role, :content, 
                :rag_context, :classification, :all_entities_json
            )
        """)
        
        with self.engine.begin() as conn:
            conn.execute(query, {
                'session_id': self.session_id,
                'user_id': self.user_id,
                'role': role,
                'content': content,
                'rag_context': rag_context,
                'classification': classification,
                'all_entities_json': entities_json
            })
    
    def get_conversation_context(self, limit: int = 3) -> str:
        """Historial SOLO de conversación (smalltalk, agradecimientos, etc)"""
        query = text("""
            SELECT role, content 
            FROM chat_history 
            WHERE session_id = :session_id 
            AND classification IN ('SMALLTALK', 'OUT_OF_SCOPE')
            ORDER BY created_at DESC 
            LIMIT :limit
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {
                'session_id': self.session_id, 
                'limit': limit
            })
            rows = list(result)[::-1]
            
            if not rows:
                return ""

            formatted_text = "CONVERSACIÓN RECIENTE:\n"
            for row in rows:
                role_name = "Usuario" if row.role == 'user' else "Asistente"
                formatted_text += f"{role_name}: {row.content}\n"
            
            return formatted_text
    
    def get_search_history(self, limit: int = 5) -> List[Dict]:
        """
        Retorna historial con TODAS las entidades de cada búsqueda.
        
        Returns:
            Lista de dicts con:
            - entity: Entidad principal (primera del array)
            - all_entities: Lista completa de entidades
            - intent: Intención
            - timestamp: Fecha
        """
        query = text("""
            SELECT 
                all_entities_json,
                classification as intent,
                created_at
            FROM chat_history 
            WHERE session_id = :session_id 
            AND classification IN ('SEARCH', 'RECOMMENDATION')
            AND all_entities_json IS NOT NULL
            ORDER BY created_at DESC 
            LIMIT :limit
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {
                'session_id': self.session_id,
                'limit': limit
            })
            
            history = []
            for row in result:
                # Deserializar all_entities_json
                all_entities = []
                primary_entity = None
                
                if row.all_entities_json:
                    try:
                        all_entities = json.loads(row.all_entities_json)
                        # La entidad principal es la primera (por prioridad del NER)
                        if all_entities and len(all_entities) > 0:
                            primary_entity = all_entities[0].get('value')
                    except json.JSONDecodeError:
                        pass
                
                history.append({
                    'entity': primary_entity,  # Principal (para compatibilidad)
                    'all_entities': all_entities,
                    'intent': row.intent,
                    'timestamp': row.created_at
                })
            
            return history
    
    def get_all_entities_from_history(self, limit: int = 3) -> List[str]:
        """
        Extrae TODAS las entidades únicas del historial reciente.
        
        Returns:
            Lista de strings con todas las entidades únicas detectadas
        """
        query = text("""
            SELECT all_entities_json
            FROM chat_history 
            WHERE session_id = :session_id 
            AND all_entities_json IS NOT NULL
            ORDER BY created_at DESC 
            LIMIT :limit
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {
                'session_id': self.session_id,
                'limit': limit
            })
            
            unique_entities = set()
            
            for row in result:
                try:
                    entities = json.loads(row.all_entities_json)
                    for ent in entities:
                        if isinstance(ent, dict) and 'value' in ent:
                            unique_entities.add(ent['value'])
                except (json.JSONDecodeError, TypeError):
                    continue
            
            return list(unique_entities)
    
    def get_entities_by_type(self, entity_type: str, limit: int = 5) -> List[str]:
        """
        Obtiene entidades de un tipo específico del historial.
        
        Args:
            entity_type: LABORATORIO, PRODUCTO, DROGA, CATEGORIA
            limit: Número de búsquedas recientes a revisar
            
        Returns:
            Lista de valores de entidades del tipo solicitado
        """
        query = text("""
            SELECT all_entities_json
            FROM chat_history 
            WHERE session_id = :session_id 
            AND all_entities_json IS NOT NULL
            ORDER BY created_at DESC 
            LIMIT :limit
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {
                'session_id': self.session_id,
                'limit': limit
            })
            
            matching_entities = []
            
            for row in result:
                try:
                    entities = json.loads(row.all_entities_json)
                    for ent in entities:
                        if isinstance(ent, dict) and ent.get('type') == entity_type:
                            matching_entities.append(ent['value'])
                except (json.JSONDecodeError, TypeError):
                    continue
            
            # Retornar sin duplicados pero manteniendo orden
            seen = set()
            unique_ordered = []
            for ent in matching_entities:
                if ent not in seen:
                    seen.add(ent)
                    unique_ordered.append(ent)
            
            return unique_ordered
    
    def get_primary_entity_from_last_search(self) -> Optional[str]:
        """
        Obtiene la entidad principal de la última búsqueda.
        
        Equivalente al antiguo get_last_entity_searched() pero extrayendo
        la primera entidad del array all_entities_json.
        
        Returns:
            String con la entidad principal o None
        """
        query = text("""
            SELECT all_entities_json
            FROM chat_history 
            WHERE session_id = :session_id 
            AND all_entities_json IS NOT NULL
            AND classification IN ('SEARCH', 'RECOMMENDATION')
            ORDER BY created_at DESC 
            LIMIT 1
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {'session_id': self.session_id})
            row = result.first()
            
            if not row:
                return None
            
            try:
                entities = json.loads(row.all_entities_json)
                if entities and len(entities) > 0:
                    return entities[0].get('value')
            except (json.JSONDecodeError, TypeError):
                pass
            
            return None
    
    def get_recent_history(self, limit: int = 5) -> str:
        """
        Historial completo formateado como texto.
        
        DEPRECADO: Usar get_conversation_context() o get_search_history()
        """
        query = text("""
            SELECT role, content 
            FROM chat_history 
            WHERE session_id = :session_id 
            ORDER BY created_at DESC 
            LIMIT :limit
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {
                'session_id': self.session_id, 
                'limit': limit
            })
            rows = list(result)[::-1]
            
            if not rows:
                return ""

            formatted_text = "HISTORIAL DE CONVERSACIÓN:\n"
            for row in rows:
                role_name = "Usuario" if row.role == 'user' else "Asistente"
                formatted_text += f"{role_name}: {row.content}\n"
            
            return formatted_text
    
    def is_new_session(self) -> bool:
        """Determina si es una sesión nueva o una continuación"""
        query = text("""
            SELECT COUNT(*) as msg_count
            FROM chat_history 
            WHERE session_id = :session_id
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {'session_id': self.session_id})
            count = result.scalar()
            return count <= 2
    
    def clear_history(self):
        """Elimina todo el historial de la sesión actual"""
        query = text("""
            DELETE FROM chat_history 
            WHERE session_id = :session_id
        """)
        
        with self.engine.begin() as conn:
            result = conn.execute(query, {
                'session_id': self.session_id
            })
            print(f"🗑️ Filas eliminadas: {result.rowcount}")


# ============================================================================
# MIGRATION SCRIPT
# ============================================================================

def migrate_remove_entity_searched_redundancy():
    """
    Script de migración para ELIMINAR la columna redundante entity_searched.
    
    ⚠️ ADVERTENCIA: Esta migración elimina datos.
    Solo ejecutar si estás seguro de que no necesitas entity_searched.
    
    Ejecutar:
    ```python
    from chat_domain.chat_history import migrate_remove_entity_searched_redundancy
    migrate_remove_entity_searched_redundancy()
    ```
    """
    engine = get_pgvector_engine()
    
    migration_sql = text("""
        -- 1. Asegurar que all_entities_json existe
        ALTER TABLE chat_history 
        ADD COLUMN IF NOT EXISTS all_entities_json TEXT;
        
        -- 2. Índice mejorado para búsquedas JSON
        CREATE INDEX IF NOT EXISTS idx_chat_history_entities_json 
        ON chat_history USING GIN ((all_entities_json::jsonb))
        WHERE all_entities_json IS NOT NULL;
        
        -- 3. OPCIONAL: Eliminar entity_searched (descomentar si estás seguro)
        -- ALTER TABLE chat_history DROP COLUMN IF EXISTS entity_searched;
        
        -- 4. Índice optimizado para búsquedas rápidas
        CREATE INDEX IF NOT EXISTS idx_chat_history_session_classification 
        ON chat_history(session_id, classification, created_at DESC)
        WHERE all_entities_json IS NOT NULL;
    """)
    
    try:
        with engine.begin() as conn:
            conn.execute(migration_sql)
        print("✅ Migración completada: redundancia eliminada")
        print("⚠️ NOTA: entity_searched NO fue eliminada por seguridad.")
        print("   Para eliminarla, descomenta la línea en el script de migración.")
    except Exception as e:
        print(f"❌ Error en migración: {e}")
        raise


def migrate_populate_all_entities_from_legacy():
    """
    OPCIONAL: Migra datos legacy de entity_searched → all_entities_json
    
    Si tienes datos viejos en entity_searched, este script los convierte
    al nuevo formato all_entities_json.
    """
    engine = get_pgvector_engine()
    
    migration_sql = text("""
        UPDATE chat_history
        SET all_entities_json = json_build_array(
            json_build_object(
                'type', 'UNKNOWN',
                'value', entity_searched,
                'position', 0
            )
        )::text
        WHERE entity_searched IS NOT NULL 
        AND all_entities_json IS NULL;
    """)
    
    try:
        with engine.begin() as conn:
            result = conn.execute(migration_sql)
        print(f"✅ Migración legacy completada: {result.rowcount} filas convertidas")
    except Exception as e:
        print(f"❌ Error en migración legacy: {e}")
        raise


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Test de funcionalidad sin redundancia
    
    history = ChatHistoryService(
        session_id="test-no-redundancy",
        user_id="user-789"
    )
    
    # Guardar búsqueda con múltiples entidades (SIN entity_searched)
    history.add_message(
        role="user",
        content="antiparasitarios de holliday y bravecto",
        classification="SEARCH",
        all_entities=[  # ← Solo este campo
            {"type": "CATEGORIA", "value": "antiparasitarios", "position": 0},
            {"type": "LABORATORIO", "value": "Holliday", "position": 23},
            {"type": "PRODUCTO", "value": "Bravecto", "position": 34}
        ]
    )
    
    # Obtener historial
    search_hist = history.get_search_history(limit=1)
    print("Historial de búsqueda:")
    print(f"  - Entidad principal: {search_hist[0]['entity']}")  # Auto-extraída
    print(f"  - Todas las entidades: {search_hist[0]['all_entities']}")
    
    # Obtener última entidad principal
    last_entity = history.get_primary_entity_from_last_search()
    print(f"\nÚltima entidad principal: {last_entity}")
    
    # Obtener todas las entidades únicas
    all_entities = history.get_all_entities_from_history(limit=3)
    print(f"Todas las entidades: {all_entities}")
    
    # Obtener solo laboratorios
    labs = history.get_entities_by_type("LABORATORIO", limit=3)
    print(f"Laboratorios: {labs}")