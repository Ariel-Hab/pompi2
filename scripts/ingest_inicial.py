#!/usr/bin/env python3
"""
Script de ingesta inicial - ACTUALIZADO CON VADEMÉCUM
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from core.db import test_pgvector_connection

# 1. IMPORTS ACTUALIZADOS
from etl_domain.extract import (
    extract_categories_from_csv,
    extract_offer_products_from_csv,
    extract_products_from_csv, 
    extract_offers_from_csv, 
    extract_companies_from_csv,
    extract_vademecum_from_csv, # <--- NECESITAS AGREGAR ESTO EN extract.py
    save_last_sync_timestamp
)
from etl_domain.enrich import (
    enrich_data_with_companies, 
    enrich_items_with_product_details, 
    enrich_products_with_categories,
    enrich_products_with_vademecum # <--- IMPORTANTE
)
from etl_domain.transform import (
    prepare_product_embeddings, 
    prepare_offer_embeddings,
    prepare_transfer_embeddings,
    prepare_company_embeddings
)
from etl_domain.load import upsert_embeddings

def filter_and_classify_offers(raw_items):
    active_offers = []
    active_transfers = []
    ignored_count = 0

    for item in raw_items:
        try:
            status = int(item.get('status', 0))
        except (ValueError, TypeError):
            status = 0
            
        if status == 1:
            active_offers.append(item)
        elif status == 3:
            item['type'] = 'transfer' 
            active_transfers.append(item)
        else:
            ignored_count += 1
            
    return active_offers, active_transfers, ignored_count

def main():
    print("=" * 60)
    print("🚀 INGESTA INICIAL - RAG Sidecar (Con Vademécum Clínico)")
    print("=" * 60)
    
    if not test_pgvector_connection():
        return 1
    
    # --- 1. EXTRAER ---
    print("\n1️⃣  Extrayendo fuentes de datos...")
    
    raw_products = extract_products_from_csv()
    
    # Limpieza de productos fantasmas
    products = [
        p for p in raw_products 
        if p.get('title') and str(p.get('title')).strip() not in ['', '0', '.', '-', 'nan', 'None']
    ]
    
    raw_offers_data = extract_offers_from_csv() 
    companies = extract_companies_from_csv()
    categories = extract_categories_from_csv()
    offer_product_links = extract_offer_products_from_csv()
    
    # NUEVO: Extracción de Vademécum
    vademecum_rows = extract_vademecum_from_csv() # <--- NUEVA FUENTE
    print(f"   📚 Vademécum cargado: {len(vademecum_rows)} registros clínicos.")

    print("\n🔍  Clasificando Offers y Transfers...")
    offers, transfers, ignored = filter_and_classify_offers(raw_offers_data)

    if not any([products, offers, transfers]):
        print("⚠️  No hay datos válidos para procesar.")
        return 0

    # --- 2. ENRIQUECER ---
    print("\n2️⃣  Enriqueciendo Metadata...")
    all_commercial_items = offers + transfers
    
    # A. Enriquecimiento Clínico (PRIORIDAD ALTA)
    if products and vademecum_rows:
        matches_vad = enrich_products_with_vademecum(products, vademecum_rows)
        print(f"   ⚕️  Datos clínicos inyectados en: {matches_vad} productos.")

    # B. Enriquecimiento Corporativo
    count_prod, count_items = enrich_data_with_companies(products, all_commercial_items, companies)
    
    # C. Enriquecimiento de Categorías
    if categories and products:
        enrich_products_with_categories(products, categories)

    # D. Propagación a Ofertas (Heredan datos clínicos del producto padre)
    count_enriched = enrich_items_with_product_details(
        all_commercial_items, 
        products, 
        offer_product_links
    )
    print(f"   ✅ Propagación completada en {count_enriched} items comerciales.")
    
    # --- 3. TRANSFORMAR ---
    print("\n3️⃣  Generando Embeddings (Con keywords expandidas)...")
    embeddings_to_load = []
    
    if products:
        # prepare_product_embeddings ahora usará los campos 'medical_indications' y 'species_data'
        # que inyectamos en el paso 2A.
        embeddings_to_load.extend(prepare_product_embeddings(products))
        
    if offers:
        embeddings_to_load.extend(prepare_offer_embeddings(offers))
        
    if transfers:
        embeddings_to_load.extend(prepare_transfer_embeddings(transfers))
        
    if companies:
        embeddings_to_load.extend(prepare_company_embeddings(companies))
    
    # --- 4. CARGAR ---
    print(f"\n4️⃣  Cargando {len(embeddings_to_load)} vectores a PGVector...")
    if embeddings_to_load:
        upsert_embeddings(embeddings_to_load)
        save_last_sync_timestamp(datetime.now())
    
    print("\n✅ PROCESO FINALIZADO EXITOSAMENTE")
    return 0

if __name__ == "__main__":
    sys.exit(main())