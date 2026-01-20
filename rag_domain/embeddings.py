"""
Gestión de modelos de embeddings
Usa FastEmbed (Local, CPU, Ligero) para no depender de RunPod ni de PyTorch pesado.
"""
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from core.config import EMBEDDING_CONFIG

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    
    if _embedding_model is None:
        print(f"⚡ Cargando FastEmbed local (modelo ligero)...")
        # Esto descargará automáticamente un modelo pequeño (~200MB) la primera vez
        # El modelo default es "BAAI/bge-small-en-v1.5", muy bueno y rápido.
        _embedding_model = FastEmbedEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            # O también: "BAAI/bge-m3" (es más pesado pero excelente)
            max_length=512
        )
        print("✅ Modelo FastEmbed listo")
    
    return _embedding_model

def embed_text(text: str) -> list[float]:
    embedder = get_embedding_model()
    return embedder.embed_query(text)