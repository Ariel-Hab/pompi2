import pytest
from etl_domain.transform import (
    expand_veterinary_terms, 
    extract_special_tags, 
    normalize_for_filter,
    product_to_text
)

# --- A. Tests de Regex y Expansión de Términos ---

@pytest.mark.parametrize("input_str, expected_substr", [
    # Rangos de peso
    ("Pipeta Perros 4kg H/10kg", "HASTA 10kg"),
    ("Pipeta Gatos H-5kg", "HASTA 5kg"),
    
    # Formas Farmacéuticas complejas (Ahora funcionarán)
    ("Amoxidal Susp. 60ml", "SUSPENSION"),
    ("Meloxicam Jga. Prellenada", "JERINGA"),
    ("Vitamina A Fco. Amp.", "FRASCO AMPOLLA"),
    ("Antiparasitario Gts.", "GOTAS"),
    
    # Laboratorios y Marcas
    ("Vacuna Zoetis Vanguard", "LABORATORIO ZOETIS"),
    ("Boehringer Ingelheim", "LABORATORIO BOEHRINGER"),
    ("John Martin", "JOHN MARTIN"),
    
    # Casos de "Falsos Positivos"
    ("Flash de camara", "Flash"), 
    ("Complejo B", "Complejo B"), 
    ("Comp. B", "COMPRIMIDOS B"), 
])
def test_regex_replacements(input_str, expected_substr):
    result = expand_veterinary_terms(input_str)
    assert expected_substr.upper() in result.upper()

# --- B. Tests de Tags Especiales (CORREGIDO) ---

def test_all_special_tags():
    """
    Valida que todas las categorías de tags se detecten.
    NOTA: Se debe expandir el término antes de buscar tags.
    """
    
    # Caso 1: Biológicos (Ya viene explícito)
    assert "vacuna" in extract_special_tags("VACUNA Defensor 3 Antirrabica")
    
    # Caso 2: Insumos (Requiere expansión previa para funcionar)
    raw_insumo = "Vetscan Rotor Kidney"
    expanded_insumo = expand_veterinary_terms(raw_insumo) 
    # Expande a: "EQUIPO DIAGNOSTICO VETSCAN..."
    assert "insumo_diagnostico" in extract_special_tags(expanded_insumo)
    
    # Caso 3: Ciclo de Vida
    assert "pediatrico" in extract_special_tags("Alimento Puppy Large")
    
    # Caso 4: Uso Hospitalario
    raw_hosp = "Propofol Hosp."
    expanded_hosp = expand_veterinary_terms(raw_hosp)
    assert "hospitalario" in extract_special_tags(expanded_hosp)
    
    # Caso 5: Trazabilidad
    raw_traz = "Ketamina Trazabilidad" # Palabra completa
    assert "trazado" in extract_special_tags(raw_traz.upper())

# --- C. Test de Generación de Texto para Embeddings ---

def test_product_to_text_structure():
    product_mock = {
        'title': 'Baytril 50mg',
        'category_name': 'Antibióticos',
        'species_data': ['Perros', 'Gatos'],
        'medical_indications': 'Infecciones bacterianas',
        'active_ingredient': 'Enrofloxacina',
        'enterprise_title': 'Bayer'
    }
    
    text = product_to_text(product_mock)
    
    assert "Antibióticos - BAYTRIL" in text 
    assert "Uso en Especies: Perros, Gatos" in text
    assert "Indicaciones Clínicas: Infecciones bacterianas" in text