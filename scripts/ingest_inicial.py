#!/usr/bin/env python3
"""
Script de ingesta inicial - VERSIÓN 4.0 MEJORADA
Vincula productos.csv (REAL) con Vademécum (CLÍNICO)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from core.db import test_pgvector_connection

# Imports actualizados
from etl_domain.extract import (
    extract_categories_from_csv,
    extract_offer_products_from_csv,
    extract_products_from_csv,  # ← productos.csv (TU BASE REAL)
    extract_offers_from_csv,
    extract_companies_from_csv,
    extract_vademecum_from_csv,  # ← Vademécum (DATOS CLÍNICOS)
    save_last_sync_timestamp
)
from etl_domain.enrich import (
    enrich_data_with_companies,
    enrich_items_with_product_details,
    enrich_products_with_categories,
    enrich_products_with_vademecum  # ← Cruce mejorado
)
from etl_domain.transform import (
    prepare_product_embeddings,
    prepare_offer_embeddings,
    prepare_transfer_embeddings,
    prepare_company_embeddings
)
from etl_domain.load import upsert_embeddings


def filter_and_classify_offers(raw_items):
    """Clasifica ofertas activas vs transfers."""
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
    print("=" * 80)
    print("🚀 INGESTA INICIAL V4.0 - productos.csv + Vademécum")
    print("=" * 80)

    if not test_pgvector_connection():
        return 1

    # ═══════════════════════════════════════════════════════════════
    # PASO 1: EXTRAER DATOS
    # ═══════════════════════════════════════════════════════════════
    print("\n1️⃣  Extrayendo fuentes de datos...")
    print("   " + "-"*76)

    # A. Productos REALES (tu base comercial)
    raw_products = extract_products_from_csv()
    print(f"   📦 productos.csv: {len(raw_products)} registros")

    # Limpieza de productos fantasmas
    products = [
        p for p in raw_products
        if p.get('title') and str(p.get('title')).strip() not in ['', '0', '.', '-', 'nan', 'None']
    ]
    print(f"   ✅ Productos válidos: {len(products)}")

    # B. Vademécum (datos clínicos)
    vademecum_rows = extract_vademecum_from_csv()
    print(f"   ⚕️  Vademécum: {len(vademecum_rows)} registros clínicos")

    # C. Otros datos
    companies = extract_companies_from_csv()
    categories = extract_categories_from_csv()
    raw_offers_data = extract_offers_from_csv()
    offer_product_links = extract_offer_products_from_csv()

    print(f"   🏭 Empresas: {len(companies)}")
    print(f"   📂 Categorías: {len(categories)}")
    print(f"   🏷️  Ofertas/Transfers: {len(raw_offers_data)}")
    print("   " + "-"*76)

    # Clasificar ofertas
    print("\n🔍  Clasificando Offers y Transfers...")
    offers, transfers, ignored = filter_and_classify_offers(raw_offers_data)
    print(f"   ✅ Ofertas: {offers}")
    print(f"   ✅ Transfers: {len(transfers)}")
    print(f"   ⏭️  Ignorados: {ignored}")

    if not any([products, offers, transfers]):
        print("⚠️  No hay datos válidos para procesar.")
        return 0

    # ═══════════════════════════════════════════════════════════════
    # PASO 2: ENRIQUECER METADATA
    # ═══════════════════════════════════════════════════════════════
    print("\n2️⃣  Enriqueciendo Metadata...")
    print("   " + "-"*76)

    all_commercial_items = offers + transfers

    # A. CRUCE PRODUCTOS ⟷ VADEMÉCUM (PRIORIDAD CRÍTICA)
    if products and vademecum_rows:
        print("\n   🔗 CRUZANDO productos.csv ⟷ Vademécum...")
        
        matched_count, unmatched = enrich_products_with_vademecum(
            products,
            vademecum_rows,
            match_threshold=0.7  # 70% similitud mínima
        )
        
        print(f"   ✅ Productos enriquecidos: {matched_count}/{len(products)}")
        
        if unmatched:
            print(f"   ⚠️  Sin datos clínicos: {len(unmatched)} productos")
            # Los productos sin match seguirán existiendo pero sin data clínica

    # B. Enriquecimiento Corporativo
    if companies:
        print("\n   🏭 Enriqueciendo con empresas...")
        count_prod, count_items = enrich_data_with_companies(
            products,
            all_commercial_items,
            companies
        )

    # C. Enriquecimiento de Categorías
    if categories and products:
        print("\n   📂 Enriqueciendo con categorías...")
        enrich_products_with_categories(products, categories)

    # D. Propagación a Ofertas (Heredan datos del producto padre)
    if all_commercial_items and products:
        print("\n   🔁 Propagando datos a ofertas/transfers...")
        count_enriched = enrich_items_with_product_details(
            all_commercial_items,
            products,
            offer_product_links
        )
        print(f"   ✅ Items enriquecidos: {count_enriched}")

    print("   " + "-"*76)

    # ═══════════════════════════════════════════════════════════════
    # PASO 3: TRANSFORMAR Y GENERAR EMBEDDINGS
    # ═══════════════════════════════════════════════════════════════
    print("\n3️⃣  Generando Embeddings...")
    print("   " + "-"*76)

    embeddings_to_load = []

    # Productos (con ID REAL de productos.csv)
    if products:
        print("\n   📄 Procesando productos...")
        prod_embeddings = prepare_product_embeddings(products)
        embeddings_to_load.extend(prod_embeddings)
        print(f"   ✅ {len(prod_embeddings)} embeddings de productos")

    # Ofertas
    if offers:
        print("\n   🏷️  Procesando ofertas...")
        offer_embeddings = prepare_offer_embeddings(offers)
        embeddings_to_load.extend(offer_embeddings)
        print(f"   ✅ {len(offer_embeddings)} embeddings de ofertas")

    # Transfers
    if transfers:
        print("\n   🎁 Procesando transfers...")
        transfer_embeddings = prepare_transfer_embeddings(transfers)
        embeddings_to_load.extend(transfer_embeddings)
        print(f"   ✅ {len(transfer_embeddings)} embeddings de transfers")

    # Empresas
    if companies:
        print("\n   🏭 Procesando empresas...")
        company_embeddings = prepare_company_embeddings(companies)
        embeddings_to_load.extend(company_embeddings)
        print(f"   ✅ {len(company_embeddings)} embeddings de empresas")

    print("   " + "-"*76)
    print(f"\n   📊 TOTAL EMBEDDINGS: {len(embeddings_to_load)}")

    # ═══════════════════════════════════════════════════════════════
    # PASO 4: CARGAR A PGVECTOR
    # ═══════════════════════════════════════════════════════════════
    print("\n4️⃣  Cargando vectores a PGVector...")
    print("   " + "-"*76)

    if embeddings_to_load:
        upsert_embeddings(embeddings_to_load)
        save_last_sync_timestamp(datetime.now())
        print(f"   ✅ {len(embeddings_to_load)} vectores cargados")
    else:
        print("   ⚠️  No hay embeddings para cargar")

    # ═══════════════════════════════════════════════════════════════
    # RESUMEN FINAL
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("✅ PROCESO COMPLETADO")
    print("="*80)
    print(f"   📦 Productos procesados: {len(products)}")
    print(f"   ⚕️  Con datos clínicos: {matched_count if matched_count else 0}")
    print(f"   🏷️  Ofertas: {len(offers)}")
    print(f"   🎁 Transfers: {len(transfers)}")
    print(f"   🏭 Empresas: {len(companies)}")
    print(f"   📊 Total vectores: {len(embeddings_to_load)}")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())