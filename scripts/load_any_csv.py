#!/usr/bin/env python3
"""
Script UNIVERSAL para cargar cualquier CSV a pgvector
Uso: python scripts/load_any_csv.py <ruta_csv> <entity_type>

Ejemplos:
  python scripts/load_any_csv.py data/products.csv product
  python scripts/load_any_csv.py data/enterprises.csv enterprise
  python scripts/load_any_csv.py data/enterprise_types.csv enterprise_type
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from core.db import test_pgvector_connection
from etl_domain.csv_parser import parse_products_csv, csv_stats
from etl_domain.transform import (
    prepare_product_embeddings, 
    prepare_offer_embeddings,
    prepare_enterprise_embeddings,
    generate_embeddings_batch
)
from etl_domain.load import upsert_embeddings
import pandas as pd


def prepare_generic_embeddings(data: list, entity_type: str) -> list:
    """
    Prepara embeddings para CUALQUIER tipo de entidad
    
    Args:
        data: Lista de diccionarios desde CSV
        entity_type: Tipo de entidad (ej: 'enterprise_type', 'category', etc)
    
    Returns:
        Lista de embeddings preparados
    """
    if not data:
        return []
    
    print(f"🔄 Transformando {len(data)} registros de tipo '{entity_type}'...")
    
    # Generar textos concatenando TODOS los campos (excepto técnicos)
    texts = []
    exclude_fields = {'id', 'timestamp', 'image', 'background_image', 'status', 'deleted_flag'}
    
    for item in data:
        parts = []
        for key, value in item.items():
            if key not in exclude_fields and value and str(value).strip():
                parts.append(f"{key}: {value}")
        
        text = ". ".join(parts) if parts else f"{entity_type} ID: {item.get('id', 'unknown')}"
        texts.append(text)
    
    # Generar embeddings
    embeddings = generate_embeddings_batch(texts)
    
    # Preparar estructura
    prepared = []
    for item, text, embedding in zip(data, texts, embeddings):
        # Metadata: guardar campos principales
        metadata = {}
        important_fields = ['title', 'name', 'code', 'abbreviation', 'description']
        
        for field in important_fields:
            if field in item and item[field]:
                metadata[field] = str(item[field])[:200]
        
        prepared.append({
            'entity_type': entity_type,
            'entity_id': int(item['id']),
            'content_text': text,
            'embedding': embedding,
            'metadata': metadata
        })
    
    print(f"✅ {len(prepared)} registros transformados")
    return prepared


def main():
    if len(sys.argv) < 3:
        print("❌ Uso: python scripts/load_any_csv.py <ruta_csv> <entity_type>")
        print("\nEjemplos:")
        print("  python scripts/load_any_csv.py data/products.csv product")
        print("  python scripts/load_any_csv.py data/enterprises.csv enterprise")
        print("  python scripts/load_any_csv.py data/enterprise_types.csv enterprise_type")
        return 1
    
    csv_path = sys.argv[1]
    entity_type = sys.argv[2]
    
    print("=" * 60)
    print(f"🚀 CARGA DE CSV - {entity_type.upper()}")
    print("=" * 60)
    
    # 1. Verificar archivo
    if not Path(csv_path).exists():
        print(f"❌ Archivo no encontrado: {csv_path}")
        return 1
    
    # 2. Mostrar stats
    print("\n📊 Analizando CSV...")
    csv_stats(csv_path)
    
    # 3. Verificar pgvector
    print("\n🔌 Verificando conexión a pgvector...")
    if not test_pgvector_connection():
        return 1
    
    # 4. Leer CSV
    print(f"\n📄 Leyendo CSV...")
    df = pd.read_csv(csv_path, encoding='utf-8')
    df = df.fillna('')
    data = df.to_dict('records')
    
    print(f"✅ Leídos {len(data)} registros")
    
    # 5. Preparar embeddings según el tipo
    print(f"\n🔄 Preparando embeddings para '{entity_type}'...")
    
    if entity_type == 'product':
        embeddings = prepare_product_embeddings(data)
    elif entity_type == 'offer':
        embeddings = prepare_offer_embeddings(data)
    elif entity_type == 'enterprise':
        embeddings = prepare_enterprise_embeddings(data)
    else:
        # Cualquier otro tipo usa el parser genérico
        embeddings = prepare_generic_embeddings(data, entity_type)
    
    # 6. Cargar
    print(f"\n💾 Cargando en pgvector...")
    upsert_embeddings(embeddings)
    
    print("\n" + "=" * 60)
    print("✅ CARGA COMPLETADA")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())