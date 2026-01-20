"""
chat.py
"""
import sys
import json
from pathlib import Path

# Configuración de path para encontrar los módulos
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chat_domain.manager import ConversationManager

def main():
    session_id = "user_session_dev"
    
    # Input del usuario
    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Tú: ")
    print(f"\n--- Chat Iniciado (Sesión: {session_id}) ---\n")

    # 1. Instanciar Manager
    manager = ConversationManager(session_id)

    # 2. Obtener respuesta COMPLETA (Diccionario)
    response_obj = manager.handle_message(user_query)

    # 3. Simular la respuesta al Frontend
    
    # A) El Texto (lo que se muestra en la burbuja de chat)
    print(f"🤖 Asistente (Texto): {response_obj['text']}")
    
    # B) El Payload (lo que usa el front para dibujar tarjetas)
    items = response_obj['data']['items']
    if items:
        print(f"\n📦 Payload para Frontend ({len(items)} objetos):")
        # Imprimimos bonito el JSON para que veas que viajan los datos
        print(json.dumps(items, indent=2, ensure_ascii=False))
    else:
        print("\n📦 Payload: (Lista vacía, no mostrar tarjetas)")

if __name__ == "__main__":
    main()