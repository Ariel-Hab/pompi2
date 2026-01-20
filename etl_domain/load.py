"""
Carga de datos en pgvector
"""
import json
from sqlalchemy import text
from core.db import get_pgvector_engine

def upsert_embeddings(data_batch: list[dict]):
    """
    Inserta o actualiza embeddings en la base de datos.
    Maneja conflictos por ID (upsert).
    """
    if not data_batch:
        return

    engine = get_pgvector_engine()

    # Corrección: Usamos sintaxis %(variable)s para todo, que es lo más seguro con psycopg2
    upsert_query = text("""
        INSERT INTO embeddings (entity_type, entity_id, content_text, embedding, metadata, created_at)
        VALUES (:entity_type, :entity_id, :content_text, :embedding, :metadata, NOW())
        ON CONFLICT (entity_type, entity_id)
        DO UPDATE SET
            content_text = EXCLUDED.content_text,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata,
            created_at = NOW();
    """)

    # Preparar datos: Postgres necesita que el vector sea un string "[1.1, 2.2]"
    # y el metadata un string JSON válido.
    clean_batch = []
    for item in data_batch:
        clean_batch.append({
            'entity_type': item['entity_type'],
            'entity_id': item['entity_id'],
            'content_text': item['content_text'],
            # Convertimos lista de floats a string "[0.1, 0.2, ...]" para pgvector
            'embedding': str(item['embedding']), 
            # Convertimos dict a string JSON para jsonb
            'metadata': json.dumps(item['metadata'])
        })

    with engine.begin() as conn:
        # SQLAlchemy maneja los :parametros automáticamente si usamos text() y conn.execute
        conn.execute(upsert_query, clean_batch)

    print(f"💾 Guardados {len(data_batch)} registros en DB.")

def get_embeddings_count():
    engine = get_pgvector_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM embeddings"))
        return result.scalar()