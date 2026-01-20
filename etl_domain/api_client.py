"""
Cliente HTTP para consumir API REST de producción
"""
import requests
from typing import List, Dict, Optional
from datetime import datetime
import time
from core.config import PRODUCTION_API_CONFIG


class ProductionAPIClient:
    """
    Cliente para interactuar con los servicios REST de producción
    """
    
    def __init__(self):
        self.base_url = PRODUCTION_API_CONFIG['base_url'].rstrip('/')
        self.token = PRODUCTION_API_CONFIG['token']
        self.timeout = PRODUCTION_API_CONFIG['timeout']
        self.retry_attempts = PRODUCTION_API_CONFIG['retry_attempts']
        
        # Headers de autenticación
        self.headers = {
            'Authorization': f'Token {self.token}',  # Ajustar según tu autenticación
            'Content-Type': 'application/json',
        }
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Realiza request HTTP con reintentos
        
        Args:
            endpoint: Path del endpoint (ej: '/products/')
            params: Query parameters opcionales
        
        Returns:
            JSON response parseado
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt < self.retry_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⚠️  Intento {attempt + 1} falló. Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Error en request después de {self.retry_attempts} intentos: {e}")
                    raise
    
    def get_products_updated_since(self, last_sync: Optional[datetime] = None) -> List[Dict]:
        """
        Obtiene productos actualizados desde una fecha
        
        Args:
            last_sync: Datetime del último sync (None = todos)
        
        Returns:
            Lista de productos
        """
        endpoint = PRODUCTION_API_CONFIG['products_endpoint']
        params = {}
        
        if last_sync:
            # Ajustar según el formato de tu API
            params['updated_after'] = last_sync.isoformat()
        
        print(f"🔌 Consultando API: {endpoint}")
        data = self._make_request(endpoint, params)
        
        # Ajustar según la estructura de respuesta de tu API
        # Puede ser data['results'], data['products'], o directamente data
        products = data.get('results', data) if isinstance(data, dict) else data
        
        print(f"📦 Obtenidos {len(products)} productos desde API")
        return products
    
    def get_offers_updated_since(self, last_sync: Optional[datetime] = None) -> List[Dict]:
        """
        Obtiene ofertas actualizadas desde una fecha
        
        Args:
            last_sync: Datetime del último sync
        
        Returns:
            Lista de ofertas
        """
        endpoint = PRODUCTION_API_CONFIG['offers_endpoint']
        params = {}
        
        if last_sync:
            params['updated_after'] = last_sync.isoformat()
        
        print(f"🔌 Consultando API: {endpoint}")
        data = self._make_request(endpoint, params)
        
        offers = data.get('results', data) if isinstance(data, dict) else data
        
        print(f"🎯 Obtenidas {len(offers)} ofertas desde API")
        return offers
    
    def test_connection(self) -> bool:
        """
        Verifica que la API esté accesible
        
        Returns:
            True si la conexión es exitosa
        """
        try:
            # Intentar un endpoint simple
            self._make_request(PRODUCTION_API_CONFIG['products_endpoint'] + '?limit=1')
            print("✅ Conexión a API de producción exitosa")
            return True
        except Exception as e:
            print(f"❌ Error conectando a API: {e}")
            return False