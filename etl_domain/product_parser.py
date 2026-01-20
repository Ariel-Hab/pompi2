"""
Extractor de metadata estructurada desde títulos de productos - MEJORADO
Parsea títulos como "APOQUEL 16 MG X 20 COMP." para extraer componentes.
"""
import re
from typing import Dict, Optional, List


class ProductMetadataExtractor:
    """
    Extrae información estructurada de títulos de productos veterinarios.
    
    MEJORAS:
    - Maneja abreviaciones complejas (OSS., SH., AMP.)
    - Extrae múltiples medidas (peso + volumen)
    - Soporta decimales con coma
    - Detecta sufijos comerciales (HOLL, HOSP, PALAT)
    """
    
    # Mapeo de abreviaciones a nombres completos
    PRESENTATION_MAP = {
        'comp': 'comprimidos',
        'comprimidos': 'comprimidos',
        'cs': 'comprimidos',  # Nuevo: "X 100 CS"
        'tab': 'comprimidos',
        'tabletas': 'comprimidos',
        'caps': 'capsulas',
        'capsulas': 'capsulas',
        'ml': 'liquido',
        'cc': 'liquido',
        'iny': 'inyectable',
        'inyectable': 'inyectable',
        'inj': 'inyectable',
        'amp': 'inyectable',  # Nuevo: "AMP. X 1,5 ML"
        'pipeta': 'pipeta',
        'pipetas': 'pipeta',
        'pip': 'pipeta',  # Nuevo
        'spot': 'pipeta',
        'sachet': 'sachet',
        'sobre': 'sachet',
        'shampoo': 'shampoo',
        'sh': 'shampoo',  # Nuevo: "OSS. SH."
        'champu': 'shampoo',
        'pomada': 'pomada',
        'crema': 'crema',
        'gel': 'gel',
        'spray': 'spray',
        'polvo': 'polvo',
        'granulado': 'granulado',
        'suspension': 'suspension',
        'susp': 'suspension',  # Nuevo
        'solucion': 'solucion',
        'sol': 'solucion',  # Nuevo
        'emulsion': 'emulsion',
        'collar': 'collar',
        'difusor': 'difusor',
        'repuesto': 'repuesto',
        'locion': 'locion',
        'atomizador': 'spray',
        'atom': 'spray',  # Nuevo
    }
    
    # Sufijos comerciales que NO son parte del nombre del producto
    COMMERCIAL_SUFFIXES = [
        'HOLL', 'HOLLIDAY', 'HOSP', 'HOSPITALARIO',
        'PALAT', 'PALATABLE', 'MASTICABLE', 'SABORIZADO',
        'PLUS', 'MAX', 'PREMIUM', 'GOLD', 'ULTRA',
        'FP',  # Fraccionado por
    ]
    
    def extract(self, title: str) -> Dict[str, any]:
        """
        Extrae metadata estructurada del título.
        
        Args:
            title: Título del producto (ej: "APOQUEL 16 MG X 20 COMP.")
            
        Returns:
            Dict con campos extraídos
        """
        if not title or not isinstance(title, str):
            return self._empty_result()
        
        title_clean = title.strip().upper()
        
        result = {
            'name': self._extract_name(title_clean),
            'dosage_value': self._extract_dosage_value(title_clean),
            'dosage_unit': self._extract_dosage_unit(title_clean),
            'weight_range': self._extract_weight_range(title_clean),
            'quantity': self._extract_quantity(title_clean),
            'volume_ml': self._extract_volume(title_clean),
            'presentation': self._extract_presentation(title_clean),
            'presentation_normalized': None,
            'commercial_suffix': self._extract_commercial_suffix(title_clean),
        }
        
        # Normalizar presentación usando el mapa
        if result['presentation']:
            result['presentation_normalized'] = self._normalize_presentation(
                result['presentation']
            )
        
        return result
    
    def _empty_result(self) -> Dict:
        """Retorna estructura vacía"""
        return {
            'name': None,
            'dosage_value': None,
            'dosage_unit': None,
            'weight_range': None,
            'quantity': None,
            'volume_ml': None,
            'presentation': None,
            'presentation_normalized': None,
            'commercial_suffix': None,
        }
    
    def _extract_name(self, title: str) -> Optional[str]:
        """
        Extrae el nombre del producto (primeras palabras antes de números).
        
        MEJORADO: Maneja abreviaciones y sufijos comerciales.
        
        Ejemplos:
            "APOQUEL 16 MG X 20 COMP." → "APOQUEL"
            "OSS. SH. PULGUICIDA GATOS X 1 LITRO" → "OSS SH PULGUICIDA"
            "PIMOCARD HOLL 1.25MG X 100 CS" → "PIMOCARD"
        """
        # Buscar hasta el primer número (que indica dosis/peso/cantidad)
        match = re.match(r'^([A-Z\.\s\-]+?)(?=\s*\d)', title)
        if match:
            name = match.group(1).strip()
            
            # Limpiar puntos de abreviaciones
            name = name.replace('.', ' ').strip()
            
            # Remover sufijos comerciales
            name_words = []
            for word in name.split():
                if word not in self.COMMERCIAL_SUFFIXES:
                    # Evitar palabras que son claramente presentación
                    if word.lower() not in ['x', 'shampoo', 'sh', 'collar', 'difusor', 'repuesto']:
                        name_words.append(word)
            
            if name_words:
                return ' '.join(name_words)
        
        # Si no hay números, tomar las primeras 2-3 palabras
        words = [w.replace('.', '').strip() for w in title.split() if w.strip()]
        if len(words) >= 2:
            # Filtrar sufijos comerciales
            filtered = [w for w in words[:3] if w not in self.COMMERCIAL_SUFFIXES]
            if filtered:
                return ' '.join(filtered[:2])
        
        return words[0] if words else None
    
    def _extract_dosage_value(self, title: str) -> Optional[float]:
        """
        Extrae el valor de la dosis en MG.
        
        MEJORADO: Soporta decimales con coma.
        
        Ejemplos:
            "APOQUEL 16 MG" → 16.0
            "PIMOCARD 1.25MG" → 1.25
            "SINCELAR 1,5 ML" → 1.5
        """
        # Patrón: número (con punto o coma) seguido de MG/ML/G
        patterns = [
            r'(\d+(?:[,\.]\d+)?)\s*MG',
            r'(\d+(?:[,\.]\d+)?)\s*G(?:\s|$)',  # Gramos
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title)
            if match:
                try:
                    # Reemplazar coma por punto para parsing
                    value_str = match.group(1).replace(',', '.')
                    return float(value_str)
                except ValueError:
                    pass
        return None
    
    def _extract_dosage_unit(self, title: str) -> Optional[str]:
        """
        Extrae la unidad de dosificación (MG, G, ML, etc).
        """
        # Buscar unidades comunes
        units = ['MG', 'G', 'ML', 'CC', 'UI', 'MCG', 'UG', 'GR']
        for unit in units:
            if re.search(rf'\d+(?:[,\.]\d+)?\s*{unit}', title):
                return unit.lower()
        return None
    
    def _extract_weight_range(self, title: str) -> Optional[str]:
        """
        Extrae rango de peso (para productos por peso del animal).
        
        MEJORADO: Maneja más variaciones.
        
        Ejemplos:
            "SIMPARICA 5 MG 1.3 A 2.5 KG" → "1.3-2.5"
            "POWER ULTRA DE 02 A 04 KG" → "2-4"
            "GATOS H/8 KG" → "0-8"
        """
        # Patrón 1: "X A Y KG"
        patterns = [
            r'(\d+(?:[,\.]\d+)?)\s*A\s*(\d+(?:[,\.]\d+)?)\s*KG',
            r'DE\s*(\d+(?:[,\.]\d+)?)\s*A\s*(\d+(?:[,\.]\d+)?)\s*KG',
            r'H/?\s*(\d+(?:[,\.]\d+)?)\s*KG',  # Nuevo: "H/8 KG" = hasta 8kg
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title)
            if match:
                if len(match.groups()) == 2:
                    min_w = float(match.group(1).replace(',', '.'))
                    max_w = float(match.group(2).replace(',', '.'))
                    return f"{min_w}-{max_w}"
                elif len(match.groups()) == 1:
                    # "H/8 KG" → "0-8"
                    max_w = float(match.group(1).replace(',', '.'))
                    return f"0-{max_w}"
        
        return None
    
    def _extract_quantity(self, title: str) -> Optional[int]:
        """
        Extrae cantidad de unidades.
        
        Ejemplos:
            "X 20 COMP" → 20
            "X 100 CS HOSP." → 100
        """
        # Patrón: X seguido de número
        match = re.search(r'X\s*(\d+)', title)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None
    
    def _extract_volume(self, title: str) -> Optional[float]:
        """
        Extrae volumen en ML.
        
        MEJORADO: Soporta decimales con coma.
        
        Ejemplos:
            "K-OTHRINA X 250 ML" → 250.0
            "SINCELAR AMP. X 1,5 ML" → 1.5
        """
        # Buscar número seguido de ML (puede estar sin X)
        match = re.search(r'(\d+(?:[,\.]\d+)?)\s*ML', title)
        if match:
            try:
                value_str = match.group(1).replace(',', '.')
                return float(value_str)
            except ValueError:
                pass
        return None
    
    def _extract_presentation(self, title: str) -> Optional[str]:
        """
        Extrae tipo de presentación del producto.
        
        MEJORADO: Detecta más abreviaciones.
        
        Ejemplos:
            "X 20 COMP." → "comp"
            "OSS. SH. PULGUICIDA" → "sh"
            "AMP. X 1,5 ML" → "amp"
        """
        title_lower = title.lower()
        
        # Buscar en el mapeo de presentaciones
        for abbrev in self.PRESENTATION_MAP.keys():
            # Buscar la palabra completa (con word boundaries)
            if re.search(rf'\b{abbrev}\.?\b', title_lower):
                return abbrev
        
        return None
    
    def _normalize_presentation(self, presentation: str) -> Optional[str]:
        """
        Normaliza la presentación usando el diccionario.
        """
        presentation_lower = presentation.lower().strip()
        return self.PRESENTATION_MAP.get(presentation_lower)
    
    def _extract_commercial_suffix(self, title: str) -> Optional[str]:
        """
        Extrae sufijos comerciales (HOLL, HOSP, etc).
        
        Ejemplos:
            "PIMOCARD HOLL 1.25MG" → "HOLL"
            "PREDNISOLONA HOLLIDAY" → "HOLLIDAY"
        """
        for suffix in self.COMMERCIAL_SUFFIXES:
            if re.search(rf'\b{suffix}\b', title):
                return suffix
        return None
    
    def build_search_terms(self, metadata: Dict) -> List[str]:
        """
        Construye términos de búsqueda adicionales basados en metadata.
        
        MEJORADO: Incluye más variaciones.
        """
        terms = []
        
        if metadata['name']:
            terms.append(metadata['name'])
        
        if metadata['dosage_value'] and metadata['dosage_unit']:
            terms.append(f"{metadata['dosage_value']}{metadata['dosage_unit']}")
        
        if metadata['presentation_normalized']:
            terms.append(metadata['presentation_normalized'])
        
        if metadata['weight_range']:
            terms.append(f"{metadata['weight_range']}kg")
        
        if metadata['volume_ml']:
            terms.append(f"{metadata['volume_ml']}ml")
        
        if metadata['commercial_suffix']:
            terms.append(metadata['commercial_suffix'].lower())
        
        return terms


# Instancia global para reutilizar
_extractor_instance = None

def get_extractor() -> ProductMetadataExtractor:
    """Retorna instancia singleton del extractor"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = ProductMetadataExtractor()
    return _extractor_instance


def extract_product_metadata(title: str) -> Dict:
    """
    Función de conveniencia para extraer metadata de un título.
    """
    extractor = get_extractor()
    return extractor.extract(title)


if __name__ == "__main__":
    # Tests con ejemplos reales del CSV
    test_titles = [
        "APOQUEL 16 MG X 20 COMP.",
        "K-OTHRINA X 250 ML.",
        "SIMPARICA 5 MG 1.3 A 2.5 KG",
        "FELIWAY CLASSIC DIFUSOR + REPUESTO 48 ML",
        "SHAMPOO ZOOVET C/CLORHEXIDINA X 350 ML",
        "POWER ULTRA DE 02 A 04 KG. X 0.7 ML",
        "OSS. SH. PULGUICIDA GATOS X 1 LITRO",
        "PIMOCARD HOLL 1.25MG X 100 CS HOSP.",
        "SINCELAR I. AMP. X 1,5 ML.",
        "APOQUEL MASTICABLE 16 MG X 20 COMP.",
    ]
    
    print("🧪 Testing Product Metadata Extractor (IMPROVED)\n")
    extractor = get_extractor()
    
    for title in test_titles:
        print(f"📦 {title}")
        metadata = extractor.extract(title)
        
        for key, value in metadata.items():
            if value is not None:
                print(f"   {key}: {value}")
        
        terms = extractor.build_search_terms(metadata)
        if terms:
            print(f"   🔍 Search terms: {', '.join(terms)}")
        
        print()