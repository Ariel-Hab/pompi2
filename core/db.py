"""
Gestión de conexiones a bases de datos
"""
from sqlalchemy import create_engine, text
from contextlib import contextmanager
import psycopg2

from core.config import get_pgvector_db_url, PGVECTOR_CONFIG


# ==================== ENGINE SQLAlchemy ====================
def get_pgvector_engine():
    """Engine para escribir/leer embeddings en pgvector"""
    return create_engine(
        get_pgvector_db_url(),
        pool_pre_ping=True,
        echo=False
    )


# ==================== CONTEXT MANAGER ====================
@contextmanager
def get_pgvector_connection():
    """Context manager para pgvector con psycopg2"""
    conn = psycopg2.connect(**PGVECTOR_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


# ==================== HEALTH CHECK ====================
def test_pgvector_connection():
    """Verifica conexión a pgvector y extensión"""
    try:
        engine = get_pgvector_engine()
        with engine.connect() as conn:
            # Verificar extensión vector
            result = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
            if result.fetchone():
                print("✅ pgvector disponible y configurado")
                return True
            else:
                print("⚠️  Extensión vector no encontrada")
                return False
    except Exception as e:
        print(f"❌ Error conectando a pgvector: {e}")
        return False