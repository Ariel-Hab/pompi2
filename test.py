import sys
import statistics
from collections import Counter
from sqlalchemy import text
from core.db import get_pgvector_engine

# ================= CONFIGURACIÓN =================
TABLE_NAME = "embeddings"

# Umbrales para alertas
MIN_PRODUCTS_PER_BRAND = 3  # Si una marca tiene menos de esto, es sospechosa (¿Typos?)
MIN_PRODUCTS_PER_CAT = 3

class Colors:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def run_integral_test():
    print(f"{Colors.HEADER}🏥 INICIANDO DIAGNÓSTICO INTEGRAL DE BASE DE DATOS{Colors.ENDC}")
    
    try:
        engine = get_pgvector_engine()
        conn = engine.connect()
        print(f"✅ Conexión establecida.\n")
    except Exception as e:
        print(f"❌ Error crítico de conexión: {e}")
        return

    # ---------------------------------------------------------
    # 1. ESTADÍSTICAS GENERALES Y SALUD DE VECTORES
    # ---------------------------------------------------------
    print(f"{Colors.BOLD}📊 1. CHEQUEO DE SALUD GENERAL{Colors.ENDC}")
    
    sql_general = text(f"""
        SELECT 
            COUNT(*) as total,
            COUNT(embedding) as with_vector,
            SUM(CASE WHEN metadata->>'title' IS NULL OR metadata->>'title' = '' THEN 1 ELSE 0 END) as missing_title,
            SUM(CASE WHEN metadata->>'enterprise_title' IS NULL THEN 1 ELSE 0 END) as missing_brand
        FROM {TABLE_NAME}
    """)
    stats = conn.execute(sql_general).fetchone()
    
    print(f"   Total Productos: {stats.total}")
    
    # Alerta Vectores
    if stats.with_vector < stats.total:
        diff = stats.total - stats.with_vector
        print(f"   ⚠️  {Colors.FAIL}HAY {diff} PRODUCTOS SIN VECTOR (EMBEDDING){Colors.ENDC}")
        print("      -> Estos productos son INVISIBLES para la búsqueda semántica.")
    else:
        print(f"   ✅ Todos los productos tienen vector.")

    # Alerta Metadata
    if stats.missing_title > 0:
        print(f"   ⚠️  {Colors.WARNING}Hay {stats.missing_title} productos SIN TÍTULO en metadata.{Colors.ENDC}")
    if stats.missing_brand > 0:
        print(f"   ⚠️  {Colors.WARNING}Hay {stats.missing_brand} productos SIN MARCA/LABORATORIO.{Colors.ENDC}")

    print("-" * 60)

    # ---------------------------------------------------------
    # 2. CONSISTENCIA DE LABORATORIOS (BRANDS)
    # ---------------------------------------------------------
    print(f"{Colors.BOLD}🏭 2. ANÁLISIS DE LABORATORIOS (Normalización){Colors.ENDC}")
    
    sql_brands = text(f"""
        SELECT metadata->>'enterprise_title' as brand, COUNT(*) as count
        FROM {TABLE_NAME}
        GROUP BY 1
        ORDER BY 2 DESC
    """)
    brands = conn.execute(sql_brands).fetchall()
    
    print(f"   Se encontraron {len(brands)} laboratorios únicos.")
    
    suspicious_brands = []
    for b_name, count in brands:
        if not b_name: continue
        if count < MIN_PRODUCTS_PER_BRAND:
            suspicious_brands.append((b_name, count))
    
    if suspicious_brands:
        print(f"   ⚠️  {Colors.WARNING}Laboratorios sospechosos (muy pocos productos, ¿posibles typos?):{Colors.ENDC}")
        for b, c in suspicious_brands[:10]:
            print(f"      - '{b}': {c} productos")
        if len(suspicious_brands) > 10: print(f"      ... y {len(suspicious_brands)-10} más.")
    else:
        print("   ✅ La distribución de laboratorios parece sana.")

    print("-" * 60)

    # ---------------------------------------------------------
    # 3. DETECCIÓN DE "FALSOS NEGATIVOS" EN OFERTAS/TRANSFERS
    # ---------------------------------------------------------
    print(f"{Colors.BOLD}🏷️  3. CONSISTENCIA DE OFERTAS Y TRANSFERS{Colors.ENDC}")
    print("   Buscando productos que DICEN ser oferta/transfer en texto pero NO tienen el flag activado...")

    # Buscamos palabras clave en el título pero flag false
    sql_ghost_transfers = text(f"""
        SELECT id, metadata->>'title' as title
        FROM {TABLE_NAME}
        WHERE 
            (metadata->>'title' ILIKE '%transfer%' OR metadata->>'title' ILIKE '%+ gift%' OR metadata->>'title' ILIKE '%bonifi%')
            AND (metadata->>'has_transfer')::boolean = false
        LIMIT 10
    """)
    ghosts = conn.execute(sql_ghost_transfers).fetchall()

    if ghosts:
        print(f"   🚩 {Colors.FAIL}POSIBLES ERRORES DE CARGA (Transfer en título, Flag False):{Colors.ENDC}")
        for g in ghosts:
            print(f"      ID: {g.id} | {g.title}")
        print("      -> Acción: Ejecutar UPDATE para poner has_transfer = true")
    else:
        print(f"   ✅ No se detectaron inconsistencias obvias en Transfers.")

    print("-" * 60)

    # ---------------------------------------------------------
    # 4. TESTEO DE FILTROS CLAVE (CATEGORÍA Y ESPECIE)
    # ---------------------------------------------------------
    print(f"{Colors.BOLD}🐶 4. DISTRIBUCIÓN DE ESPECIES Y CATEGORÍAS{Colors.ENDC}")
    
    sql_species = text(f"""
        SELECT metadata->>'species_filter', COUNT(*) 
        FROM {TABLE_NAME} 
        GROUP BY 1 ORDER BY 2 DESC LIMIT 5
    """)
    species_dist = conn.execute(sql_species).fetchall()
    
    print("   Top Especies:")
    for sp, c in species_dist:
        print(f"      - {sp}: {c}")

    # Verificar vacíos
    sql_empty_species = text(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE metadata->>'species_filter' IS NULL OR metadata->>'species_filter' = ''")
    empty_sp = conn.execute(sql_empty_species).scalar()
    
    if empty_sp > 0:
         print(f"   ⚠️  {Colors.FAIL}HAY {empty_sp} PRODUCTOS SIN ESPECIE DEFINIDA.{Colors.ENDC}")
    
    print("-" * 60)

    # ---------------------------------------------------------
    # 5. TEST DE PESOS (Weight Range)
    # ---------------------------------------------------------
    print(f"{Colors.BOLD}⚖️  5. FORMATO DE PESOS (Weight Range){Colors.ENDC}")
    # Buscamos formatos rotos (no nulos, pero que no siguen patrón numérico)
    # Nota: Esto es una validación simple en Python
    
    sql_weights = text(f"SELECT id, metadata->>'weight_range' FROM {TABLE_NAME} WHERE metadata->>'weight_range' IS NOT NULL LIMIT 50")
    weights = conn.execute(sql_weights).fetchall()
    
    bad_weights = 0
    for row in weights:
        w_str = row[1]
        # Esperamos "min-max" o "val"
        if w_str and not any(char.isdigit() for char in w_str):
            bad_weights += 1
            if bad_weights <= 5:
                print(f"      Mal formato detectado ID {row[0]}: '{w_str}'")

    if bad_weights > 0:
        print(f"   ⚠️  {Colors.WARNING}Se detectaron {bad_weights} rangos de peso con formato extraño.{Colors.ENDC}")
    else:
        print("   ✅ Muestra de pesos verificada correctamente.")

    print(f"\n{Colors.HEADER}🏁 DIAGNÓSTICO FINALIZADO{Colors.ENDC}")
    conn.close()

if __name__ == "__main__":
    run_integral_test()