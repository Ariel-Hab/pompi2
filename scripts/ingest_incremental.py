#!/usr/bin/env python3
"""
Script de ingesta incremental desde API REST
Sincroniza solo registros nuevos/actualizados desde servicios de producción
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from core.db import test_pgvector_connection
from etl_domain.extract import (
    extract_products_from_api,
    extract_offers_from_api,
    get_last_sync_timestamp,
    save_last_sync_timestamp
)
from etl_domain.api_client import ProductionAPIClient
from etl_domain.transform import prepare_product_embeddings, prepare_offer_embeddings
from etl_domain.load import upsert_embeddings


def main():
    print("=" * 60)
    print("🔄 INGESTA INCREMENTAL DESDE API - RAG Sidecar")
    print("=" * 60)
    
    # 1. Verificar conexiones
    print("\n1️⃣  Verificando conexiones...")
    if not test_pgvector_connection():
        return 1
    
    # Verificar API
    api_client = ProductionAPIClient()
    if not api_client.test_connection():
        print("❌ No se pudo conectar a la API de producción")
        return 1
    
    # 2. Obtener timestamp del último sync
    print("\n2️⃣  Verificando última sincronización...")
    last_sync = get_last_sync_timestamp()
    
    if not last_sync:
        print("⚠️  No hay registro de sync previo. Ejecuta ingest_initial.py primero.")
        return 1
    
    print(f"   Último sync: {last_sync}")
    
    # 3. Extraer solo registros nuevos desde API
    print("\n3️⃣  Consultando API de producción...")
    products = extract_products_from_api(last_sync=last_sync)
    offers = extract_offers_from_api(last_sync=last_sync)
    
    if not products and not offers:
        print("✅ No hay registros nuevos para sincronizar")
        return 0
    
    # 4. Transformar
    print("\n4️⃣  Generando embeddings...")
    product_embeddings = prepare_product_embeddings(products)
    offer_embeddings = prepare_offer_embeddings(offers)
    
    # 5. Cargar
    print("\n5️⃣  Actualizando pgvector...")
    if product_embeddings:
        upsert_embeddings(product_embeddings)
    if offer_embeddings:
        upsert_embeddings(offer_embeddings)
    
    # 6. Actualizar timestamp
    current_time = datetime.now()
    save_last_sync_timestamp(current_time)
    
    print("\n" + "=" * 60)
    print("✅ INGESTA INCREMENTAL COMPLETADA")
    print("=" * 60)
    print(f"  • Productos sincronizados: {len(products)}")
    print(f"  • Ofertas sincronizadas: {len(offers)}")
    print(f"  • Nuevo timestamp: {current_time}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())