"""
QUERY OPTIMIZER V4 - LLM-Driven Intent & Hybrid Search
El LLM es la autoridad final para el Intent, asistido por el NER.
"""
import sys
import json
from pathlib import Path
from typing import Dict, Optional, List

# Asumimos que los archivos están en la estructura del proyecto
from rag_domain.ner_classifier import ClassificationResult, VeterinaryNERClassifier

# Import config desde el proyecto
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from ai_gateway.llm_client import LLMService
except ImportError:
    print("⚠️ [OPTIMIZER] No se pudo importar LLMService, usando mock")
    class LLMService:
        def generate(self, system_prompt, user_prompt):
            return "SEARCH"

class QueryOptimizer:
    """
    Optimizador de Queries V4.
    
    ARQUITECTURA:
    1. NER: Extrae datos duros (entidades) para filtros SQL.
    2. LLM: Decide la intención (Intent) usando el texto + las entidades del NER como contexto.
    3. Hybrid Output: Vector (texto) + Filtros (NER) + Intent (LLM).
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        self.ner = VeterinaryNERClassifier(data_dir=data_dir)
        
        try:
            self.ai = LLMService()
            self.has_llm = True
        except Exception as e:
            print(f"⚠️ [OPTIMIZER] LLM no disponible: {e}")
            self.ai = None
            self.has_llm = False
    
    def optimize(self, raw_text: str) -> Dict:
        print(f"\n🔍 [OPTIMIZER] Processing: '{raw_text}'")
        
        # 1. NER (Retorna lista plana de entidades)
        classification = self.ner.classify(raw_text)
        
        # 2. Intent (LLM)
        final_intent = self._determine_intent_via_llm(raw_text, classification)
        
        # 3. Filtros (Usamos todas las entidades detectadas)
        detected_filters = self._build_search_filters(classification)
        
        # 4. Vector Search Text (Query completo limpio)
        vector_search_text = self._prepare_search_text(raw_text, final_intent)

        return {
            "intent": final_intent,
            "search_input": vector_search_text,
            "search_filters": detected_filters,
            # ❌ ELIMINAR: "main_entity": "",
            "debug_ner": [e.entity_value for e in classification.all_entities]
        }

    def _build_search_filters(self, classification: ClassificationResult) -> Dict:
        """
        Construye diccionario de filtros usando TODAS las entidades detectadas
        y metadatos estructurados del NER.
        """
        filters = {}
        
        # --- 1. FILTROS BOOLEANOS Y ESTRUCTURADOS ---
        # Ofertas y Transfers
        if classification.filters.get('is_offer'): filters['is_offer'] = True
        if classification.filters.get('is_transfer'): filters['is_transfer'] = True
        
        # Presentación (Ej: 'comprimidos', 'inyectable') - El NER ya lo normaliza
        if classification.filters.get('presentation'):
            filters['presentation'] = classification.filters['presentation']

        # --- 2. ENTIDADES (Listas completas por Tipo) ---
        
        # Laboratorios
        labs = [e.entity_value for e in classification.all_entities if e.entity_type == "LABORATORIO"]
        if labs: filters['laboratorios'] = labs 

        # Categorías (Ej: Alimento, Farmacia)
        cats = [e.entity_value for e in classification.all_entities if e.entity_type == "CATEGORIA"]
        if cats: filters['categorias'] = cats

        # Especies (Ej: Perros, Gatos)
        species = [e.entity_value for e in classification.all_entities if e.entity_type == "ESPECIE"]
        if species: filters['species'] = species

        # Productos Específicos (Ej: Bravecto)
        prods = [e.entity_value for e in classification.all_entities if e.entity_type == "PRODUCTO"]
        if prods: filters['target_products'] = prods

        # --- NUEVOS AGREGADOS ---
        
        # Drogas (Ej: Amoxicilina, Ivermectina)
        drugs = [e.entity_value for e in classification.all_entities if e.entity_type == "DROGA"]
        if drugs: filters['drogas'] = drugs

        # Acciones Terapéuticas (Ej: Antibiótico, Antiparasitario)
        actions = [e.entity_value for e in classification.all_entities if e.entity_type == "ACCION"]
        if actions: filters['acciones'] = actions

        # Conceptos (Ej: Pipeta, Collar, Talco)
        concepts = [e.entity_value for e in classification.all_entities if e.entity_type == "CONCEPTO"]
        if concepts: filters['conceptos'] = concepts

        # --- 3. VALORES NUMÉRICOS ---
        # Dosis (Ej: 20mg)
        if classification.filters.get('dosage_value'):
            filters['dosage_value'] = classification.filters['dosage_value']
            filters['dosage_unit'] = classification.filters.get('dosage_unit')
        
        return filters

    def _determine_intent_via_llm(self, raw_text: str, classification: ClassificationResult) -> str:
        """
        Usa el LLM para decidir la intención, dándole las entidades detectadas como 'pistas'.
        """
        # Fallback si no hay LLM configurado
        if not self.has_llm:
            return classification.intent # Fallback al heurístico del NER

        # Formatear las entidades para que el LLM las entienda fácil
        entities_context = "\n".join([
            f"- {e.entity_type}: {e.entity_value}" 
            for e in classification.all_entities
        ])
        
        if not entities_context:
            entities_context = "(Ninguna entidad detectada)"

        system_prompt = """
        Eres el clasificador de intención del Asistente Virtual de Rincón Transfer (Distribuidora Farmacéutica Veterinaria).
        Tu objetivo es distinguir si el usuario busca información del catálogo/ofertas o si es una interacción social.

        TUS CATEGORÍAS:
        1. SEARCH: Consultas sobre el catálogo, drogas, laboratorios, o beneficios comerciales (Transfers/Ofertas). 
        Ej: "¿Qué transfers hay de Zoetis?", "Bravecto info", "Drogas para la tos".
        2. RECOMMENDATION: Consultas clínicas donde el usuario busca asesoramiento sobre qué producto usar para un cuadro médico.
        Ej: "Tengo un perro con sarna, ¿qué me recomendás?".
        3. SMALLTALK: Saludos, presentaciones personales ("Soy X"), agradecimientos o charlas casuales.
        4. OUT_OF_SCOPE: Temas que no pertenecen al rubro veterinario o institucional de Rincón Transfer.

        REGLAS DE ORO DE IDENTIDAD:
        - PRIORIDAD DE NOMBRE: Si el usuario dice "Hola, soy [Nombre]", es SMALLTALK. Ignora si el nombre coincide con un laboratorio o producto en las "Entidades Detectadas".
        - FOCO INSTITUCIONAL: SEARCH no es solo "comprar", es "informarse". Si pregunta por una oferta o un laboratorio, es SEARCH.
        - EL LLM MANDA: Las entidades del NER son solo una referencia. Si el mensaje no tiene estructura de consulta técnica o comercial, trátalo como SMALLTALK.
        """

        user_prompt = f"""
        MENSAJE DEL USUARIO: "{raw_text}"
        
        ENTIDADES DETECTADAS POR NER (Pistas técnicas):
        {entities_context}
        
        Responde solo con la categoría:
        """

        try:
            # Llamada al LLM
            response = self.ai.generate(system_prompt, user_prompt)
            intent = response.strip().upper().replace(".", "")
            
            # Validación de seguridad
            valid_intents = {'SEARCH', 'RECOMMENDATION', 'SMALLTALK', 'OUT_OF_SCOPE'}
            if intent not in valid_intents:
                print(f"⚠️ [LLM] Respuesta no estándar '{intent}', forzando SEARCH")
                return "SEARCH"
                
            return intent
            
        except Exception as e:
            print(f"❌ [LLM] Error detectando intent: {e}")
            return classification.intent # Fallback al NER si el LLM explota

    
    def _prepare_search_text(self, raw_text: str, intent: str) -> str:
        """
        Limpia el texto para la búsqueda vectorial y FTS.
        """
        if intent == "SMALLTALK":
            return ""  # ✅ No buscar nada para smalltalk
            
        # Limpiar stop phrases
        clean = raw_text.strip().lower()
        
        for phrase in ['busco', 'necesito', 'quiero', 'tenes', 'tienen', 'mostrame', 
                    'dame', 'que productos', 'precio de', 'info de']:
            clean = clean.replace(phrase, "")
        
        # Limpiar signos
        clean = re.sub(r'[?¿!¡]', '', clean).strip()
        
        return clean

if __name__ == "__main__":
    # Test rápido
    opt = QueryOptimizer()
    
    # Simulamos que tenemos un LLM conectado (o usará el mock que devuelve SEARCH)
    queries = [
        "hola buenos dias",                 # LLM debería decir SMALLTALK
        "mi perro se rasca mucho",          # LLM debería decir RECOMMENDATION
        "precio del bravecto 20kg",         # LLM debería decir SEARCH (NER aporta: Bravecto)
        "que me recomendas para pulgas"     # LLM debería decir RECOMMENDATION
    ]
    
    print("\n⚡ TEST RÁPIDO DE OPTIMIZER V4 (LLM-DRIVEN) ⚡")
    for q in queries:
        res = opt.optimize(q)
        print("-" * 40)