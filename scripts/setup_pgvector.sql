-- Configuración inicial de pgvector
-- Este script se ejecuta automáticamente al crear el contenedor

-- Habilitar extensión vector
CREATE EXTENSION IF NOT EXISTS vector;

-- Crear tabla de embeddings
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,
    entity_id INTEGER NOT NULL,
    content_text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,  -- Dimensión del modelo
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraint única para evitar duplicados
    UNIQUE(entity_type, entity_id)
);

-- Índice para búsquedas vectoriales (IVFFlat)
-- NOTA: Este índice se crea DESPUÉS de tener datos (ver comentario abajo)
-- CREATE INDEX embeddings_vector_idx ON embeddings 
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- Índices para filtros comunes
CREATE INDEX IF NOT EXISTS idx_entity_type ON embeddings(entity_type);
CREATE INDEX IF NOT EXISTS idx_created_at ON embeddings(created_at DESC);

-- Comentario sobre el índice IVFFlat:
-- IVFFlat requiere datos para entrenar. Ejecutar DESPUÉS de la carga inicial:
-- 1. Cargar datos con ingest_initial.py
-- 2. Luego ejecutar manualmente:
--    CREATE INDEX embeddings_vector_idx ON embeddings 
--    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);