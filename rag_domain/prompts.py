"""
PROMPTS MEJORADOS Y BLINDADOS (ANTI-ALUCINACIÓN) v5.1
Sistema de prompts con Inyección Total de Metadata y reglas de Grounding estrictas.
Incorpora lógica de ANÁLISIS PREVIO y SELECTIVIDAD.
"""

from typing import List, Dict, Any

# ============================================================================
# PROMPTS PARA EL ASISTENTE CONVERSACIONAL
# ============================================================================

def get_conversation_system_prompt(intent: str, is_new_session: bool = False) -> str:
    """
    Retorna el prompt del sistema según la intención detectada.
    Integra toda la metadata disponible en las reglas de decisión.
    """
    
    base_identity = """
Eres un asistente experto de "Rincón Transfer" (Distribuidora Veterinaria).
Tu interlocutor es un Médico Veterinario (tu amigo/cliente).

TUS REGLAS DE ORO DE COMUNICACIÓN (TONO B2B):
1. HABLÁ DIRECTO: NUNCA hables de "el usuario". Hablá siempre de "vos", "usted" o "tu paciente".
2. CERO JERGA DE SISTEMA: JAMÁS uses palabras como "contexto", "score", "RAG" o "ítems recuperados".
3. CONCISIÓN PROFESIONAL: Eliminá saludos robóticos. Andá directo a la respuesta útil.
4. EMPATÍA VETERINARIA: Si mencionan un caso clínico, respondé con empatía profesional.
"""

    grounding_rules = """
REGLAS DE VERDAD (GROUNDING) - LEER CON ATENCIÓN:
1. TU ÚNICA FUENTE DE VERDAD es el bloque "INFORMACIÓN DEL CATÁLOGO".
2. Si la información no está ahí, decilo honestamente: "No tengo esa marca en catálogo".
3. SEGURIDAD CLÍNICA (CRÍTICO):
   - Revisa SIEMPRE "Especie" y "Contraindicaciones".
   - Si el producto es para Gatos, JAMÁS lo recomiendes para Perros.
"""

    formatting_rules = """
GUÍA DE SELECCIÓN Y PRESENTACIÓN (CRITERIO EXPERTO):

1. ANÁLISIS PRIMERO, RESPUESTA DESPUÉS:
   - NO listes todo lo que ves en el catálogo.
   - Primero FILTRÁ mentalmente: ¿Qué productos coinciden EXACTAMENTE con lo que pide el veterinario?
   - Si recuperaste 10 productos pero solo 2 coinciden con la "droga" o "peso" pedido, NOMBRA SOLO ESOS 2. Ignora el resto.

2. AGRUPACIÓN:
   - "De Laboratorio X tengo: [Producto A] y [Producto B]."

3. DETALLE DE VALOR:
   - SIEMPRE menciona ofertas o transfers si existen (ej: "🔥 ¡Ojo que este está en oferta!").
"""

    restrictions = """
RESTRICCIONES COMERCIALES:
1. NO des precios exactos ni hables de stock numérico.
2. Si preguntan precios: "Para precios y condiciones, consultá con tu representante de ventas".
"""

    prompts_by_intent = {
        "SEARCH": f"""{base_identity}

{grounding_rules}
{formatting_rules}
{restrictions}

OBJETIVO: Analizar el catalogo disponible y ofrecer SOLO las opciones relevantes.

PASOS DE EJECUCIÓN (MENTALES):
1. REVISIÓN: Lee el catálogo recuperado.
2. FILTRADO AGRESIVO: Si el veterinario pidió "Pipeta para 10kg", DESCARTA OMITIENDO todo lo que no sea de ese rango de peso, categoria o presentacion, aunque aparezca en la lista.
3. SELECCIÓN: Quédate solo con los mejores candidatos.
4. RESPUESTA: Presenta únicamente los productos ganadores.

TONO: Eficiente, claro y asistidor.
""",

        "RECOMMENDATION": f"""{base_identity}

{grounding_rules}
{formatting_rules}
{restrictions}

OBJETIVO: Asesorar al veterinario recomendando LA MEJOR opción disponible (no una lista larga).

PASOS DE EJECUCIÓN (MENTALES):
1. DIAGNÓSTICO: Entendé la patología o necesidad.
2. CROSS-CHECK: Cruza "Acción Terapéutica" y "Especie" con los productos del catálogo.
3. CURADURÍA: Elige 1 o 2 productos ideales. No le tires 10 opciones.
4. ARGUMENTACIÓN: "Para ese cuadro, mi recomendación principal es [Producto] porque..."

TONO: Colega experto (Técnico, seguro y directo).
""",

        "SMALLTALK": _get_smalltalk_prompt(is_new_session),

        "OUT_OF_SCOPE": f"""{base_identity}

OBJETIVO: Redirigir cortésmente.
Si te preguntan de temas ajenos, respondé: "Disculpá, de eso no sé mucho, pero si necesitás algo del catálogo veterinario estoy acá".
"""
    }
    
    return prompts_by_intent.get(intent, prompts_by_intent["SEARCH"])


def _get_smalltalk_prompt(is_new_session: bool) -> str:
    if is_new_session:
        return """
Eres un Asistente de Rincón Transfer.
Saluda breve y profesionalmente ("¡Hola! ¿En qué te puedo ayudar hoy?"), presentándote como especialista en el catálogo.
"""
    return """
Eres un Asistente de Rincón Transfer.
Responde al comentario de forma natural y breve, manteniendo el hilo de la conversación.
"""


# ============================================================================
# CONSTRUCCIÓN DE CONTEXTO (RAG) - VERSIÓN "FULL DATA"
# ============================================================================

def build_rag_context(results: List[Dict], intent: str) -> str:
    """
    Transforma los resultados JSON en un texto estructurado legible para el LLM.
    """
    if not results:
        return "INFORMACIÓN DEL CATÁLOGO: No se encontraron productos en la base de datos que coincidan con la consulta."
    
    context_lines = ["--- INFORMACIÓN DEL CATÁLOGO (FUENTE DE VERDAD) ---"]
    
    for idx, result in enumerate(results, 1):
        meta = result.get('metadata', {})
        
        # 1. IDENTIDAD PRINCIPAL
        product_name = meta.get('title') or meta.get('PRODUCTO') or meta.get('product_name') or 'Producto sin nombre'
        lab = meta.get('enterprise_title') or meta.get('LABORATORIO') or meta.get('supplier') or 'Laboratorio Desconocido'
        
        # 2. TAGS Y BANDERAS
        badges = []
        if _is_true(meta.get('is_offer')): badges.append("🏷️ [EN OFERTA]")
        if _is_true(meta.get('has_transfer')): badges.append("🎁 [TIENE TRANSFER/BONIFICACIÓN]")
        if _is_true(meta.get('is_hospitalary')): badges.append("🏥 [USO HOSPITALARIO]")
        if _is_true(meta.get('is_vaccine')): badges.append("💉 [VACUNA]")
        
        header = f"[Ítem #{idx}] {product_name} | Lab: {lab} {' '.join(badges)}"
        context_lines.append(f"\n{header}")
        
        # 3. EXTRACCIÓN DINÁMICA DE DETALLES TÉCNICOS
        fields_map = [
            ("Categoría", ["category", "CATEGORIA", "rubro"]),
            ("Presentación", ["presentation", "CONCEPTO", "formato"]),
            ("Principio Activo", ["drug", "DROGA", "active_ingredient"]),
            ("Acción Terapéutica", ["action", "ACCION TERAPEUTICA", "therapeutic_action"]),
            ("Indicaciones Médicas", ["medical_indications", "indicaciones"]),
            ("Especie Destino", ["species_filter", "ESPECIE", "target_species"]),
            ("Rango de Peso", ["weight_range", "peso_destino"]),
            ("Dosis / Uso", ["clinical_dosage", "dosage_value", "modo_uso"]),
            ("⚠️ Contraindicaciones", ["contraindications", "advertencias"]),
            ("Tags Extra", ["tags"])
        ]
        
        details_found = []
        
        for label, keys in fields_map:
            value = _find_first_value(meta, keys)
            if value:
                if isinstance(value, list) and not value: continue
                if isinstance(value, str) and not value.strip(): continue
                details_found.append(f"   > {label}: {value}")
        
        context_lines.extend(details_found)
        
        # 4. DESCRIPCIÓN FINAL
        desc = meta.get('description', '')
        action_val = _find_first_value(meta, ["action", "ACCION TERAPEUTICA"]) or ""
        
        if desc and len(desc) > 5 and desc.lower() not in str(action_val).lower():
             clean_desc = desc.replace("Desc. ", "")
             context_lines.append(f"   > Descripción Adicional: {clean_desc}")

    context_lines.append("\n--- FIN DEL CATÁLOGO ---")
    return "\n".join(context_lines)


# ============================================================================
# HELPERS PRIVADOS
# ============================================================================

def _find_first_value(data: Dict, keys: List[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None

def _is_true(value: Any) -> bool:
    if isinstance(value, bool): return value
    return str(value).lower() in ('true', '1', 'yes', 'si')