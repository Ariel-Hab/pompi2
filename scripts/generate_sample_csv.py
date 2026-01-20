#!/usr/bin/env python3
"""
Helper para generar CSVs de ejemplo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etl_domain.csv_parser import generate_sample_csv, csv_stats
from core.config import ETL_CONFIG

if __name__ == "__main__":
    print("🔧 Generando archivos CSV de ejemplo...")
    generate_sample_csv()
    
    print("\n" + "=" * 60)
    csv_stats(ETL_CONFIG['csv_products_path'])
    print("=" * 60)
    csv_stats(ETL_CONFIG['csv_offers_path'])
    
    print("\n✅ Listo! Edita los archivos CSV en /app/data/ con tus datos reales")