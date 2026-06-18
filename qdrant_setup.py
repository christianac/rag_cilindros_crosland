#!/usr/bin/env python3
"""
Script de inicialización para base de datos Qdrant RAG
Especializado en clasificación de imágenes de cilindros
Configurado para usar Qdrant Cloud (servicio en línea)

Vector size: 768 dimensiones (text-embedding-004 de Google Gemini)
"""

import os
import logging
from typing import Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantCylinderDB:
    def __init__(self, 
                 cloud_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 host: str = "localhost", 
                 port: int = 6333):
        """
        Inicializar conexión con Qdrant Cloud
        
        Args:
            cloud_url: URL completa de Qdrant Cloud (ej: https://xxx.qdrant.io)
            api_key: API Key de Qdrant Cloud (requerido para cloud)
            host: Host de Qdrant local (alternativa)
            port: Puerto de Qdrant local (alternativa)
        """
        self.collection_name = "cylinder_images"
        # Dimensión oficial de text-embedding-004 (Google Gemini)
        # Referencia: https://ai.google.dev/gemini-api/docs/models#text-embedding
        self.vector_size = 768
        
        # Priorizar Qdrant Cloud si se proporciona URL
        self.cloud_url = cloud_url or os.getenv("QDRANT_CLOUD_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.host = host
        self.port = port
        
        # Conectar a Qdrant
        try:
            if self.cloud_url and self.api_key:
                # Conexión a Qdrant Cloud
                self.client = QdrantClient(url=self.cloud_url, api_key=self.api_key)
                logger.info(f"Conectado a Qdrant Cloud: {self.cloud_url}")
            else:
                # Conexión local como fallback
                self.client = QdrantClient(host=self.host, port=self.port)
                logger.info(f"Conectado a Qdrant local: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Error conectando a Qdrant: {e}")
            raise
    
    def create_collection(self) -> bool:
        """
        Crear colección para imágenes de cilindros
        
        Returns:
            bool: True si la colección se creó o ya existe
        """
        try:
            # Verificar si la colección ya existe
            collections = self.client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if self.collection_name in existing_collections:
                logger.info(f"La colección '{self.collection_name}' ya existe")
                return True
            
            # Crear nueva colección
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"Colección '{self.collection_name}' creada exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error creando colección: {e}")
            return False
    
    def get_collection_info(self) -> Dict:
        """
        Obtener información de la colección
        
        Returns:
            Dict: Información de la colección
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.name,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count
            }
        except Exception as e:
            logger.error(f"Error obteniendo información de colección: {e}")
            return {}
    
    def delete_collection(self) -> bool:
        """
        Eliminar colección (usar con cuidado)
        
        Returns:
            bool: True si se eliminó exitosamente
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Colección '{self.collection_name}' eliminada")
            return True
        except Exception as e:
            logger.error(f"Error eliminando colección: {e}")
            return False
    
    def test_connection(self) -> bool:
        """
        Probar la conexión con Qdrant
        
        Returns:
            bool: True si la conexión es exitosa
        """
        try:
            collections = self.client.get_collections()
            logger.info(f"Conexión exitosa. Colecciones existentes: {len(collections.collections)}")
            return True
        except Exception as e:
            logger.error(f"Error en conexión: {e}")
            return False

def main():
    """
    Función principal para inicializar la base de datos
    """
    print("=== Inicialización de Base de Datos Qdrant RAG ===")
    print("Especializada en clasificación de imágenes de cilindros")
    print("Configurado para Qdrant Cloud\n")
    
    # Cargar configuración desde variables de entorno
    QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    
    # Validar configuración
    if not QDRANT_CLOUD_URL:
        print("⚠️  ADVERTENCIA: No se encontró QDRANT_CLOUD_URL en variables de entorno")
        print("   Configurando para usar Qdrant local como fallback")
        QDRANT_CLOUD_URL = None
    
    if QDRANT_CLOUD_URL and not QDRANT_API_KEY:
        print("❌ ERROR: Se proporcionó QDRANT_CLOUD_URL pero falta QDRANT_API_KEY")
        return False
    
    if QDRANT_CLOUD_URL:
        print(f"Conectando a Qdrant Cloud: {QDRANT_CLOUD_URL}")
    else:
        print(f"Conectando a Qdrant local: localhost:6333")
    
    try:
        # Inicializar base de datos
        db = QdrantCylinderDB(
            cloud_url=QDRANT_CLOUD_URL,
            api_key=QDRANT_API_KEY
        )
        
        # Probar conexión
        if not db.test_connection():
            print("❌ No se pudo conectar a Qdrant")
            return False
        
        print("✅ Conexión exitosa")
        
        # Crear colección
        print("\nCreando colección para imágenes de cilindros...")
        if db.create_collection():
            print("✅ Colección creada/verificada")
        else:
            print("❌ Error creando colección")
            return False
        
        # Mostrar información de la colección
        print("\nInformación de la colección:")
        info = db.get_collection_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\n🎉 Base de datos inicializada correctamente")
        print("\nConfiguración utilizada:")
        if QDRANT_CLOUD_URL:
            print(f"  - Tipo: Qdrant Cloud")
            print(f"  - URL: {QDRANT_CLOUD_URL}")
        else:
            print(f"  - Tipo: Qdrant local")
        print(f"  - Dimensión de vector: {db.vector_size} "
              f"(text-embedding-004 · google-genai SDK)")

        print("\nEstructura de metadatos que se utilizará:")
        print("  - cylinder_condition:  'correct' | 'dented' | 'false_positive'")
        print("  - confidence_score:    puntuación de confianza (0.0 – 1.0)")
        print("  - upload_timestamp:    timestamp de subida")
        print("  - source:              origen del dato (n8n, manual, training…)")
        print("  - verified:            si fue verificado por un humano")
        print("  - description:         descripción generada por gemini-2.5-flash")
        print("  - embedding_model:     text-embedding-004")
        print("  - vision_model:        gemini-2.5-flash")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante inicialización: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)