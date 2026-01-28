"""
Weight Extractor - Sistema Multi-Capa
Extracción robusta de rangos de peso desde títulos de productos
"""
import re
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class WeightRange:
    """Representa un rango de peso con metadata."""
    min_weight: float
    max_weight: float
    confidence: float  # 0.0 - 1.0
    method: str  # 'explicit', 'keyword', 'family', 'standard'
    raw_text: str = ""


class WeightExtractor:
    """
    Sistema multi-capa para extracción de rangos de peso.
    
    Estrategias (en orden de prioridad):
    1. Rango explícito: "10-20kg", "10 a 20kg" (100% confiable)
    2. Keywords: "hasta 20kg", "desde 10kg" (100% confiable)
    3. Familia de productos: Consulta DB (90% confiable)
    4. Rangos estándar: Convención veterinaria (85% confiable)
    """
    
    def __init__(self, db_connection=None):
        self.db = db_connection
        
        # Rangos estándar en productos veterinarios
        self.standard_ranges = [
            (0, 4, 'muy_pequeno'),
            (4, 10, 'pequeno'),
            (10, 20, 'mediano'),
            (20, 40, 'grande'),
            (40, 60, 'muy_grande'),
            (60, 100, 'gigante')
        ]
    
    def extract(
        self, 
        title: str, 
        product_family: Optional[str] = None
    ) -> Optional[WeightRange]:
        """
        Extrae rango de peso usando múltiples estrategias.
        
        Args:
            title: Título del producto
            product_family: Nombre base del producto (ej: "Bravecto")
        
        Returns:
            WeightRange con min, max, confidence y método usado
        """
        
        # ═══════════════════════════════════════════════════════════
        # ESTRATEGIA 1: Rango explícito (máxima prioridad)
        # ═══════════════════════════════════════════════════════════
        explicit = self._extract_explicit_range(title)
        if explicit:
            return explicit
        
        # ═══════════════════════════════════════════════════════════
        # ESTRATEGIA 2: Keywords (hasta, desde, máximo, mínimo)
        # ═══════════════════════════════════════════════════════════
        keyword = self._extract_with_keywords(title)
        if keyword:
            return keyword
        
        # ═══════════════════════════════════════════════════════════
        # ESTRATEGIA 3: Familia de productos (consulta DB)
        # ═══════════════════════════════════════════════════════════
        if product_family and self.db:
            family = self._infer_from_family(product_family, title)
            if family:
                return family
        
        # ═══════════════════════════════════════════════════════════
        # ESTRATEGIA 4: Rangos estándar veterinarios
        # ═══════════════════════════════════════════════════════════
        single_weight = self._extract_single_weight(title)
        if single_weight:
            standard = self._infer_standard_range(single_weight, title)
            if standard:
                return standard
        
        # No se pudo extraer peso
        return None
    
    def _extract_explicit_range(self, title: str) -> Optional[WeightRange]:
        """
        Detecta rangos explícitos: "10-20kg", "10 a 20kg".
        Confidence: 100%
        """
        
        title_clean = title.lower().strip()
        
        # ═══════════════════════════════════════════════════════════
        # PATRÓN 1: Con guion (10-20kg, 10kg-20kg, 10-20 kg)
        # ═══════════════════════════════════════════════════════════
        pattern1 = r'(\d+(?:[.,]\d+)?)\s*(?:kg)?\s*[-–]\s*(\d+(?:[.,]\d+)?)\s*(?:kg|kilos?)'
        match1 = re.search(pattern1, title_clean)
        
        if match1:
            num1 = float(match1.group(1).replace(',', '.'))
            num2 = float(match1.group(2).replace(',', '.'))
            
            # Auto-corregir orden si está invertido
            min_w = min(num1, num2)
            max_w = max(num1, num2)
            
            return WeightRange(
                min_weight=min_w,
                max_weight=max_w,
                confidence=1.0,
                method='explicit_range',
                raw_text=match1.group(0)
            )
        
        # ═══════════════════════════════════════════════════════════
        # PATRÓN 2: Con "a" (10 a 20kg, 5 a 10 kilos)
        # ═══════════════════════════════════════════════════════════
        pattern2 = r'(\d+(?:[.,]\d+)?)\s*(?:kg)?\s+[aA]\s+(\d+(?:[.,]\d+)?)\s*(?:kg|kilos?)'
        match2 = re.search(pattern2, title_clean)
        
        if match2:
            num1 = float(match2.group(1).replace(',', '.'))
            num2 = float(match2.group(2).replace(',', '.'))
            
            min_w = min(num1, num2)
            max_w = max(num1, num2)
            
            return WeightRange(
                min_weight=min_w,
                max_weight=max_w,
                confidence=1.0,
                method='explicit_range_a',
                raw_text=match2.group(0)
            )
        
        return None
    
    def _extract_with_keywords(self, title: str) -> Optional[WeightRange]:
        """
        Detecta usando palabras clave: hasta, desde, máximo, mínimo.
        Confidence: 100%
        """
        
        title_clean = title.lower().strip()
        
        # Keywords para límite superior
        max_keywords = [
            'hasta', 'maximo', 'max', 'máximo',
            'no más de', 'no mas de', 'menor a', 'menor que'
        ]
        
        # Keywords para límite inferior
        min_keywords = [
            'desde', 'minimo', 'min', 'mínimo',
            'apartir', 'a partir', 'mayor a', 'mayor que',
            'mas de', 'más de'
        ]
        
        max_weight = None
        min_weight = None
        raw_text = ""
        
        # Buscar límite superior
        for kw in max_keywords:
            pattern = f'{kw}\\s+(\\d+(?:[.,]\\d+)?)\\s*(?:kg|kilos?)'
            match = re.search(pattern, title_clean)
            if match:
                max_weight = float(match.group(1).replace(',', '.'))
                raw_text = match.group(0)
                break
        
        # Buscar límite inferior
        for kw in min_keywords:
            pattern = f'{kw}\\s+(\\d+(?:[.,]\\d+)?)\\s*(?:kg|kilos?)'
            match = re.search(pattern, title_clean)
            if match:
                min_weight = float(match.group(1).replace(',', '.'))
                if raw_text:
                    raw_text += " " + match.group(0)
                else:
                    raw_text = match.group(0)
                break
        
        # Si encontró al menos uno
        if max_weight is not None or min_weight is not None:
            return WeightRange(
                min_weight=min_weight if min_weight is not None else 0.0,
                max_weight=max_weight if max_weight is not None else 999.0,
                confidence=1.0,
                method='keyword',
                raw_text=raw_text
            )
        
        return None
    
    def _extract_single_weight(self, title: str) -> Optional[float]:
        """
        Extrae peso único: "20kg", "10 kilos".
        """
        
        title_clean = title.lower().strip()
        
        pattern = r'(\d+(?:[.,]\d+)?)\s*(?:kg|kilos?)'
        match = re.search(pattern, title_clean)
        
        if match:
            return float(match.group(1).replace(',', '.'))
        
        return None
    
    def _infer_from_family(
        self, 
        product_family: str, 
        title: str
    ) -> Optional[WeightRange]:
        """
        Infiere consultando otros productos de la misma familia.
        Confidence: 90%
        
        Ejemplo:
        - Título: "BRAVECTO 10KG"
        - Busca en DB: "BRAVECTO 4-10KG", "BRAVECTO 10-20KG"
        - Si "10" coincide con límite superior de "4-10KG" → (4, 10)
        """
        
        if not self.db:
            return None
        
        # Extraer peso del título actual
        weight = self._extract_single_weight(title)
        if not weight:
            return None
        
        try:
            # Buscar productos relacionados con rangos explícitos
            from sqlalchemy import text
            
            query = text("""
                SELECT metadata->>'title' as title,
                       metadata->>'weight_value' as weight_value
                FROM embeddings
                WHERE lower(metadata->>'title') ILIKE :pattern
                  AND metadata->>'title' ~ '\\d+-\\d+\\s*kg'
                LIMIT 20
            """)
            
            results = self.db.execute(
                query, 
                {'pattern': f'%{product_family.lower()}%'}
            ).fetchall()
            
            # Extraer rangos conocidos de la familia
            for row in results:
                product_title = row['title']
                
                # Extraer rango del título
                range_match = re.search(
                    r'(\d+(?:[.,]\d+)?)\s*-\s*(\d+(?:[.,]\d+)?)\s*kg', 
                    product_title.lower()
                )
                
                if range_match:
                    min_w = float(range_match.group(1).replace(',', '.'))
                    max_w = float(range_match.group(2).replace(',', '.'))
                    
                    # Si el peso coincide con el límite superior (±10%)
                    if abs(max_w - weight) / max_w < 0.1:
                        return WeightRange(
                            min_weight=min_w,
                            max_weight=max_w,
                            confidence=0.9,
                            method='family_inference',
                            raw_text=f"Inferido de familia {product_family}"
                        )
                    
                    # Si el peso coincide con el límite inferior (±10%)
                    if abs(min_w - weight) / min_w < 0.1:
                        return WeightRange(
                            min_weight=min_w,
                            max_weight=max_w,
                            confidence=0.85,
                            method='family_inference',
                            raw_text=f"Inferido de familia {product_family}"
                        )
        
        except Exception as e:
            print(f"⚠️ [WeightExtractor] Error en family inference: {e}")
        
        return None
    
    def _infer_standard_range(
        self, 
        weight: float, 
        title: str
    ) -> Optional[WeightRange]:
        """
        Infiere rango basado en convenciones veterinarias.
        Confidence: 85%
        
        Asume que el peso mencionado es típicamente el LÍMITE SUPERIOR
        (convención común: "Bravecto de 10" = "para perros hasta 10kg")
        """
        
        # Estrategia 1: Coincidencia exacta con límite superior de rango estándar
        for min_w, max_w, size_name in self.standard_ranges:
            if abs(weight - max_w) < 0.1:  # Coincidencia exacta
                return WeightRange(
                    min_weight=min_w,
                    max_weight=max_w,
                    confidence=0.85,
                    method='standard_range_upper',
                    raw_text=f"Rango estándar: {size_name}"
                )
        
        # Estrategia 2: El peso está dentro de un rango estándar
        for min_w, max_w, size_name in self.standard_ranges:
            if min_w < weight <= max_w:
                return WeightRange(
                    min_weight=min_w,
                    max_weight=max_w,
                    confidence=0.75,  # Menor confidence
                    method='standard_range_contains',
                    raw_text=f"Rango estándar: {size_name}"
                )
        
        # Estrategia 3: Peso fuera de rangos estándar
        # Asumir que es límite superior, inferir límite inferior
        if weight > 100:
            # Muy grande, probablemente límite superior
            return WeightRange(
                min_weight=60.0,
                max_weight=weight,
                confidence=0.6,
                method='standard_range_fallback',
                raw_text="Inferido: rango grande"
            )
        elif weight < 4:
            # Muy pequeño, probablemente límite superior
            return WeightRange(
                min_weight=0.0,
                max_weight=weight,
                confidence=0.6,
                method='standard_range_fallback',
                raw_text="Inferido: rango muy pequeño"
            )
        
        # No se pudo inferir con confianza
        return None


# ═══════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════

def test_weight_extraction():
    """Suite de tests para validar extracción de pesos."""
    
    extractor = WeightExtractor()
    
    test_cases = [
        # Rangos explícitos (confidence 1.0)
        ("POWER GOLD 10-20KG PIPETA", (10.0, 20.0), 1.0, 'explicit'),
        ("BRAVECTO 20-40KG", (20.0, 40.0), 1.0, 'explicit'),
        ("ADVOCATE 4-10 KG", (4.0, 10.0), 1.0, 'explicit'),
        ("SIMPARICA 5KG-10KG", (5.0, 10.0), 1.0, 'explicit'),
        
        # Rangos con "A" (confidence 1.0)
        ("NEXGARD 10 A 20 KG", (10.0, 20.0), 1.0, 'explicit'),
        ("FRONTLINE 5 A 10KG", (5.0, 10.0), 1.0, 'explicit'),
        
        # Rangos invertidos (auto-corrección)
        ("PRODUCTO 40-20KG", (20.0, 40.0), 1.0, 'explicit'),
        ("WEIRD 60-40 KILOS", (40.0, 60.0), 1.0, 'explicit'),
        
        # Keywords (confidence 1.0)
        ("PRODUCTO HASTA 20KG", (0.0, 20.0), 1.0, 'keyword'),
        ("PRODUCTO DESDE 10KG", (10.0, 999.0), 1.0, 'keyword'),
        ("MAXIMO 40 KG", (0.0, 40.0), 1.0, 'keyword'),
        ("MINIMO 5 KILOS", (5.0, 999.0), 1.0, 'keyword'),
        
        # Pesos únicos → Inferencia estándar (confidence 0.85)
        ("BRAVECTO 10KG", (4.0, 10.0), 0.85, 'standard'),
        ("SIMPARICA 20KG", (10.0, 20.0), 0.85, 'standard'),
        ("NEXGARD 40KG", (20.0, 40.0), 0.85, 'standard'),
        ("ADVOCATE 4KG", (0.0, 4.0), 0.85, 'standard'),
        
        # Pesos intermedios (dentro de rango)
        ("PRODUCTO 15KG", (10.0, 20.0), 0.75, 'standard'),
        ("PRODUCTO 8KG", (4.0, 10.0), 0.75, 'standard'),
        
        # Sin peso
        ("POWER ULTRA PIPETA", None, 0.0, None),
        ("ANTIBIOTICO X 250 ML", None, 0.0, None),
    ]
    
    print("\n" + "="*70)
    print("🧪 TEST SUITE: Weight Extraction System")
    print("="*70 + "\n")
    
    passed = 0
    failed = 0
    
    for idx, (title, expected_range, expected_conf, expected_method) in enumerate(test_cases, 1):
        result = extractor.extract(title)
        
        # Verificar resultado
        if result is None:
            actual_range = None
            actual_conf = 0.0
            actual_method = None
        else:
            actual_range = (result.min_weight, result.max_weight)
            actual_conf = result.confidence
            actual_method = result.method.split('_')[0]  # Simplificar método
        
        # Comparar
        range_match = actual_range == expected_range
        conf_match = abs(actual_conf - expected_conf) < 0.01 if expected_conf else True
        method_match = (expected_method in actual_method) if expected_method and actual_method else (expected_method == actual_method)
        
        success = range_match and conf_match and method_match
        
        if success:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"{status} #{idx:2d} '{title[:50]}'")
        
        if result:
            print(f"         → Range: {actual_range} (conf: {actual_conf:.2f}, method: {result.method})")
        else:
            print(f"         → Range: None")
        
        if not success:
            print(f"         → Expected: {expected_range} (conf: {expected_conf:.2f})")
            if not range_match:
                print(f"         → ❌ Range mismatch!")
            if not conf_match:
                print(f"         → ❌ Confidence mismatch!")
            if not method_match:
                print(f"         → ❌ Method mismatch! (expected: {expected_method})")
        
        print()
    
    print("="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Success rate: {passed/(passed+failed)*100:.1f}%")
    print("="*70 + "\n")
    
    return passed, failed


if __name__ == "__main__":
    # Ejecutar tests
    passed, failed = test_weight_extraction()
    
    # Ejemplos de uso
    print("\n" + "="*70)
    print("📝 Ejemplos de Uso")
    print("="*70 + "\n")
    
    extractor = WeightExtractor()
    
    examples = [
        "POWER GOLD 10-20KG PIPETA",
        "BRAVECTO 10KG",
        "PRODUCTO HASTA 40 KILOS",
        "SIMPARICA 5 A 10KG",
    ]
    
    for title in examples:
        result = extractor.extract(title)
        
        if result:
            print(f"Título: '{title}'")
            print(f"  → Rango: {result.min_weight}-{result.max_weight}kg")
            print(f"  → Confidence: {result.confidence:.0%}")
            print(f"  → Método: {result.method}")
            print()
        else:
            print(f"Título: '{title}'")
            print(f"  → Sin peso detectado")
            print()