import csv
import os
from typing import List, Dict, Any, Optional
from unidecode import unidecode

class AttachmentBuilder:
    def __init__(self):
        self.BASE_MEDIA_URL = "http://www.integhra.com/media/"
        self.UI_CONFIG = [
            {"type": "product", "path": "app/data/products.csv"},
            {"type": "offer",   "path": "app/data/offers.csv"}
        ]
        self.LLM_SOURCE_CONFIG = [
            {"type": "product", "path": "app/data/vademecum.csv"}, 
            {"type": "offer",   "path": "app/data/offers.csv"}     
        ]

    def _normalize_strict(self, text_val: Any) -> str:
        """Normalización para matchear nombres de productos."""
        if not text_val: return ""
        clean = unidecode(str(text_val)).lower()
        for char in [' ', '.', '-', ',', '/', '(', ')', '"', "'"]:
            clean = clean.replace(char, '')
        return clean.strip()

    def enrich_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Hidrata candidatos con información del Vademecum."""
        if not candidates: return []
        rich_data_map = self._load_csv_data(candidates, config_list=self.LLM_SOURCE_CONFIG)

        for cand in candidates:
            e_type = cand.get('entity_type', 'product')
            e_id = str(cand.get('entity_id', ''))
            key = f"{e_type}_{e_id}"
            meta = cand.get('metadata', {})

            if key in rich_data_map:
                meta.update(rich_data_map[key])
                meta['_source_origin'] = "VADEMECUM_ENRICHED" if e_type == 'product' else "OFFER_ENRICHED"
            else:
                meta['_source_origin'] = "UNKNOWN"
                meta['_warning'] = "NOT_FOUND_IN_CSV"
        return candidates

    def build_attachments(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Crea tarjetas para el frontend usando Products.csv."""
        if not candidates: return []
        commercial_data_map = self._load_csv_data(candidates, config_list=self.UI_CONFIG)
        attachments = []
        for item in candidates:
            key = f"{item.get('entity_type')}_{item.get('entity_id')}"
            if key in commercial_data_map:
                attach = self._map_row_to_attachment(item.get('entity_type'), commercial_data_map[key])
                if attach: attachments.append(attach)
        return attachments

    def _load_csv_data(self, candidates: List[Dict], config_list: List[Dict] = None) -> Dict[str, Dict]:
        active_config = config_list if config_list else self.UI_CONFIG
        data_map = {}
        target_ids = {str(c.get('entity_id')) for c in candidates}
        target_titles = {self._normalize_strict(c.get('metadata', {}).get('title', '')) for c in candidates}

        for config in active_config:
            path = config["path"]
            if not os.path.exists(path): path = path.replace("app/", "")
            if not os.path.exists(path): continue

            is_vademecum = "vademecum" in path.lower()

            try:
                with open(path, mode='r', encoding='utf-8-sig', errors='replace') as f:
                    # Detectamos el delimitador automáticamente (en tu caso es ',')
                    reader = csv.DictReader(f)
                    for row in reader:
                        clean_row = {str(k).strip(): str(v).strip() for k, v in row.items() if k}
                        
                        if is_vademecum:
                            # MODIFICACIÓN: Buscamos específicamente en la columna 'PRODUCTO' de tu CSV
                            row_title = clean_row.get('PRODUCTO') or clean_row.get('title')
                            if row_title:
                                norm_row_title = self._normalize_strict(row_title)
                                if norm_row_title in target_titles:
                                    for c in candidates:
                                        if self._normalize_strict(c['metadata'].get('title', '')) == norm_row_title:
                                            data_map[f"product_{c.get('entity_id')}"] = clean_row
                        else:
                            # Búsqueda comercial por ID
                            row_id = clean_row.get('ID') or clean_row.get('id')
                            if row_id and str(row_id) in target_ids:
                                data_map[f"{config['type']}_{row_id}"] = clean_row
            except Exception as e:
                print(f"Error en {path}: {e}")
        return data_map

    def _map_row_to_attachment(self, e_type: str, row: Dict) -> Optional[Dict]:
        if e_type == 'product':
            return {
                "type": "product",
                "data": {
                    "id": row.get('ID') or row.get('id'),
                    "title": row.get('PRODUCTO') or row.get('title'),
                    "selling_price": row.get('PR LISTA', '0.0'),
                    "image": f"{self.BASE_MEDIA_URL}{row.get('Link institucional', '')}"
                }
            }
        return None

# ============================================================================
# TEST PARA VALIDAR TU CSV ESPECÍFICO
# ============================================================================
if __name__ == "__main__":
    builder = AttachmentBuilder()
    
    # 1. Simulamos un candidato con el nombre exacto de tu CSV
    mock_candidates = [{
        "entity_id": 335656, 
        "entity_type": "product",
        "metadata": {"title": "AMOX PLUS 100 MG. X 50 COMP"} # Nombre exacto de tu CSV
    }]
    
    print("\n🔍 Validando Vademecum con tus headers...")
    enriched = builder.enrich_candidates(mock_candidates)
    
    meta = enriched[0]['metadata']
    if meta.get('_source_origin') == "VADEMECUM_ENRICHED":
        print("✅ ÉXITO: Producto encontrado por nombre 'PRODUCTO'.")
        print(f"   Dato recuperado: {meta.get('ACCION TERAPEUTICA')}")
        print(f"   Dosificación: {meta.get('Dosificación')[:50]}...")
    else:
        print("❌ FALLO: No se encontró. Revisa que el nombre del archivo sea vademecum.csv")