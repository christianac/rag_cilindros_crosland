#!/usr/bin/env python3
"""
Procesador de imágenes para base de datos RAG de cilindros
Utiliza Google Gemini para generar embeddings multimodales
"""

import os
import io
import base64
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
from PIL import Image
import numpy as np
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import google.generativeai as genai
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CylinderImageProcessor:
    def __init__(self, 
                 gemini_api_key: Optional[str] = None,
                 qdrant_cloud_url: Optional[str] = None,
                 qdrant_api_key: Optional[str] = None,
                 qdrant_host: str = "localhost", 
                 qdrant_port: int = 6333):
        """
        Inicializar el procesador de imágenes con Gemini y Qdrant Cloud
        
        Args:
            gemini_api_key: API Key de Google Gemini
            qdrant_cloud_url: URL de Qdrant Cloud
            qdrant_api_key: API Key de Qdrant Cloud
            qdrant_host: Host de Qdrant local (fallback)
            qdrant_port: Puerto de Qdrant local (fallback)
        """
        self.collection_name = "cylinder_images"
        
        # Configurar Gemini
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY es requerido. Obtén una en https://aistudio.google.com/app/apikey"
            )
        
        genai.configure(api_key=self.gemini_api_key)
        
        # Configurar modelo Gemini
        self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
        self.embedding_model = 'models/text-embedding-004'  # Para descripciones
        self.vector_size = 768  # Dimensión de embeddings Gemini
        
        # Conectar a Qdrant (priorizar Cloud)
        self.qdrant_cloud_url = qdrant_cloud_url or os.getenv("QDRANT_CLOUD_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        
        try:
            if self.qdrant_cloud_url and self.qdrant_api_key:
                self.qdrant_client = QdrantClient(
                    url=self.qdrant_cloud_url,
                    api_key=self.qdrant_api_key
                )
                logger.info(f"Conectado a Qdrant Cloud: {self.qdrant_cloud_url}")
            else:
                self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
                logger.info(f"Conectado a Qdrant local: {qdrant_host}:{qdrant_port}")
        except Exception as e:
            logger.error(f"Error conectando a Qdrant: {e}")
            raise
        
        # Categorías de clasificación
        self.categories = {
            "correct": "Cilindro en buen estado, sin daños visibles",
            "dented": "Cilindro con abolladuras o daños evidentes",
            "false_positive": "Cilindro que parece tener daños pero está en buen estado"
        }
        
        logger.info("Procesador de imágenes inicializado con Gemini")
    
    def encode_image_to_base64(self, image: Union[str, Image.Image, bytes]) -> str:
        """
        Convertir imagen a base64
        
        Args:
            image: Imagen (ruta, objeto PIL, o bytes)
            
        Returns:
            str: Imagen codificada en base64
        """
        try:
            if isinstance(image, str):
                with open(image, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            elif isinstance(image, bytes):
                return base64.b64encode(image).decode('utf-8')
            elif isinstance(image, Image.Image):
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG')
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            else:
                raise ValueError("Formato de imagen no soportado")
        except Exception as e:
            logger.error(f"Error codificando imagen: {e}")
            raise
    
    def get_image_description(self, image: Union[str, Image.Image, bytes]) -> str:
        """
        Obtener descripción detallada de la imagen usando Gemini Vision
        
        Args:
            image: Imagen a describir
            
        Returns:
            str: Descripción generada por Gemini
        """
        try:
            # Preparar imagen
            if isinstance(image, str):
                pil_image = Image.open(image).convert("RGB")
            elif isinstance(image, bytes):
                pil_image = Image.open(io.BytesIO(image)).convert("RGB")
            elif isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
            else:
                raise ValueError("Formato de imagen no soportado")
            
            # Prompt para análisis detallado
            prompt = """Analiza esta imagen de un cilindro industrial en detalle. 
            Describe específicamente:
            1. El estado general del cilindro
            2. Si tiene abolladuras, golpes o deformaciones visibles
            3. Si la superficie parece dañada o en buen estado
            4. Color, forma y características visibles
            5. Cualquier detalle relevante sobre su condición
            
            Responde en español de forma concisa y técnica."""
            
            # Generar descripción
            response = self.vision_model.generate_content([prompt, pil_image])
            description = response.text.strip()
            
            logger.debug(f"Descripción generada: {description[:100]}...")
            return description
            
        except Exception as e:
            logger.error(f"Error generando descripción: {e}")
            return f"Error generando descripción: {str(e)}"
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generar embedding vectorial de un texto usando Gemini
        
        Args:
            text: Texto para generar embedding
            
        Returns:
            np.ndarray: Vector embedding
        """
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            embedding = np.array(result['embedding'], dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            raise
    
    def process_image_for_rag(self, image: Union[str, Image.Image, bytes]) -> Tuple[np.ndarray, str]:
        """
        Procesar imagen completa: generar descripción y embedding
        
        Args:
            image: Imagen a procesar
            
        Returns:
            Tuple[np.ndarray, str]: (embedding, descripción)
        """
        try:
            # Generar descripción con Gemini Vision
            logger.info("Generando descripción con Gemini Vision...")
            description = self.get_image_description(image)
            
            # Generar embedding de la descripción
            logger.info("Generando embedding...")
            embedding = self.generate_embedding(description)
            
            logger.info(f"Embedding generado: {embedding.shape}")
            return embedding, description
            
        except Exception as e:
            logger.error(f"Error procesando imagen: {e}")
            raise
    
    def upload_image(self, 
                    image: Union[str, Image.Image, bytes],
                    cylinder_condition: str,
                    confidence_score: float = 1.0,
                    source: str = "manual",
                    verified: bool = True,
                    additional_metadata: Optional[Dict] = None) -> str:
        """
        Subir imagen a la base de datos RAG
        
        Args:
            image: Imagen a subir
            cylinder_condition: Estado del cilindro ('correct', 'dented', 'false_positive')
            confidence_score: Puntuación de confianza (0.0 - 1.0)
            source: Fuente de la imagen
            verified: Si ha sido verificado por humano
            additional_metadata: Metadatos adicionales
            
        Returns:
            str: ID del punto insertado
        """
        try:
            # Validar categoría
            if cylinder_condition not in self.categories:
                raise ValueError(f"Categoría inválida. Debe ser una de: {list(self.categories.keys())}")
            
            # Procesar imagen (descripción + embedding)
            logger.info("Procesando imagen con Gemini...")
            embedding, description = self.process_image_for_rag(image)
            
            # Validar dimensión del embedding
            if embedding.shape[0] != self.vector_size:
                logger.warning(
                    f"Dimensión del embedding ({embedding.shape[0]}) "
                    f"difiere de la configurada ({self.vector_size})"
                )
            
            # Crear ID único
            point_id = str(uuid.uuid4())
            
            # Preparar metadatos
            metadata = {
                "cylinder_condition": cylinder_condition,
                "condition_description": self.categories[cylinder_condition],
                "description": description,
                "confidence_score": float(confidence_score),
                "upload_timestamp": datetime.now().isoformat(),
                "source": source,
                "verified": verified,
                "embedding_model": "Gemini text-embedding-004",
                "vision_model": "Gemini 1.5 Flash",
                "vector_size": int(embedding.shape[0])
            }
            
            # Agregar metadatos adicionales si se proporcionan
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Crear punto para insertar
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=metadata
            )
            
            # Insertar en Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Imagen subida exitosamente con ID: {point_id}")
            logger.info(f"Condición: {cylinder_condition}, Confianza: {confidence_score}")
            
            return point_id
            
        except Exception as e:
            logger.error(f"Error subiendo imagen: {e}")
            raise
    
    def search_similar_images(self, 
                             query_image: Union[str, Image.Image, bytes],
                             limit: int = 5,
                             score_threshold: float = 0.7,
                             filter_condition: Optional[str] = None) -> List[Dict]:
        """
        Buscar imágenes similares en la base de datos
        
        Args:
            query_image: Imagen de consulta
            limit: Número máximo de resultados
            score_threshold: Umbral mínimo de similitud
            filter_condition: Filtrar por condición específica
            
        Returns:
            List[Dict]: Lista de imágenes similares con metadatos
        """
        try:
            # Generar embedding de la imagen de consulta
            logger.info("Procesando imagen de consulta...")
            query_embedding, _ = self.process_image_for_rag(query_image)
            
            # Preparar filtros si se especifican
            query_filter = None
            if filter_condition:
                query_filter = {
                    "must": [
                        {
                            "key": "cylinder_condition",
                            "match": {"value": filter_condition}
                        }
                    ]
                }
            
            # Realizar búsqueda
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Formatear resultados
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "metadata": result.payload
                })
            
            logger.info(f"Encontradas {len(results)} imágenes similares")
            return results
            
        except Exception as e:
            logger.error(f"Error buscando imágenes similares: {e}")
            raise
    
    def classify_cylinder(self, 
                         image: Union[str, Image.Image, bytes],
                         confidence_threshold: float = 0.8) -> Dict:
        """
        Clasificar estado de cilindro basado en imágenes similares
        
        Args:
            image: Imagen a clasificar
            confidence_threshold: Umbral de confianza para clasificación
            
        Returns:
            Dict: Resultado de clasificación
        """
        try:
            # Obtener descripción de la imagen
            description = self.get_image_description(image)
            
            # Buscar imágenes similares
            similar_images = self.search_similar_images(
                query_image=image,
                limit=10,
                score_threshold=0.5
            )
            
            if not similar_images:
                return {
                    "predicted_condition": "unknown",
                    "confidence": 0.0,
                    "description": description,
                    "reason": "No se encontraron imágenes similares en la base de datos"
                }
            
            # Analizar condiciones de imágenes similares
            condition_votes = {"correct": 0, "dented": 0, "false_positive": 0}
            total_score = 0
            
            for result in similar_images:
                condition = result["metadata"]["cylinder_condition"]
                score = result["score"]
                conf = result["metadata"].get("confidence_score", 1.0)
                weight = score * conf
                
                condition_votes[condition] += weight
                total_score += weight
            
            # Determinar condición predicha
            predicted_condition = max(condition_votes, key=condition_votes.get)
            confidence = condition_votes[predicted_condition] / total_score if total_score > 0 else 0
            
            # Clasificar como confiable si supera el umbral
            is_confident = confidence >= confidence_threshold
            
            return {
                "predicted_condition": predicted_condition,
                "confidence": float(confidence),
                "is_confident": is_confident,
                "description": description,
                "condition_scores": condition_votes,
                "similar_images_count": len(similar_images),
                "best_match_score": similar_images[0]["score"] if similar_images else 0
            }
            
        except Exception as e:
            logger.error(f"Error clasificando cilindro: {e}")
            raise

# Función auxiliar para procesar desde n8n
def process_image_from_n8n(image_data: str, 
                          cylinder_condition: str,
                          confidence_score: float = 1.0) -> Dict:
    """
    Función específica para procesar imágenes desde n8n
    
    Args:
        image_data: Imagen codificada en base64
        cylinder_condition: Estado del cilindro
        confidence_score: Puntuación de confianza
        
    Returns:
        Dict: Resultado del procesamiento
    """
    try:
        # Decodificar imagen base64
        if image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Inicializar procesador
        processor = CylinderImageProcessor()
        
        # Subir imagen
        point_id = processor.upload_image(
            image=image_bytes,
            cylinder_condition=cylinder_condition,
            confidence_score=confidence_score,
            source="n8n",
            verified=False
        )
        
        return {
            "success": True,
            "point_id": point_id,
            "message": "Imagen procesada exitosamente con Gemini"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Error procesando imagen"
        }

if __name__ == "__main__":
    # Ejemplo de uso
    try:
        processor = CylinderImageProcessor()
        print("=== Procesador de Imágenes de Cilindros ===")
        print("Configurado con Google Gemini y Qdrant Cloud")
        print(f"Modelo de embeddings: {processor.embedding_model}")
        print(f"Modelo de visión: gemini-1.5-flash")
    except Exception as e:
        print(f"Error inicializando procesador: {e}")