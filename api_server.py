#!/usr/bin/env python3
"""
API Server para conectar n8n con el procesador de imágenes RAG
SDK:    google-genai (nueva)
Visión: gemini-2.5-flash
Embed:  gemini-embedding-2 (3072d)
DB:     Qdrant Cloud
"""

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from dotenv import load_dotenv
from image_processor import CylinderImageProcessor, process_image_from_n8n

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar Flask
app = Flask(__name__)
CORS(app)

# Inicializar procesador de imágenes
try:
    processor = CylinderImageProcessor(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        qdrant_cloud_url=os.getenv("QDRANT_CLOUD_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY")
    )
    logger.info("Procesador inicializado | gemini-2.5-flash + gemini-embedding-2 (3072d) + Qdrant Cloud")
except Exception as e:
    logger.error(f"Error inicializando procesador: {e}")
    processor = None

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de salud"""
    if processor is None:
        return jsonify({
            "status": "error", 
            "message": "Procesador no disponible",
            "endpoints": []
        }), 500
    
    # Verificar configuración
    config_info = {
        "qdrant_type":     "cloud" if processor.qdrant_cloud_url else "local",
        "embedding_model": processor.embedding_model,
        "vision_model":    processor.vision_model_id,
        "vector_size":     processor.vector_size,
        "sdk":             "google-genai",
    }

    return jsonify({
        "status": "healthy",
        "message": "API funcionando con google-genai SDK + gemini-2.5-flash",
        "config": config_info,
        "endpoints": [
            "/process-image",
            "/classify-image", 
            "/search-similar",
            "/stats",
            "/health"
        ]
    })

@app.route('/process-image', methods=['POST'])
def process_image():
    """
    Procesar y almacenar imagen de cilindro
    
    Body JSON:
    {
        "image_data": "base64_encoded_image",
        "cylinder_condition": "correct|dented|false_positive",
        "confidence_score": 0.95,
        "source_info": {...}
    }
    """
    try:
        if processor is None:
            return jsonify({"success": False, "error": "Procesador no disponible"}), 500
        
        # Validar datos de entrada
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No se proporcionaron datos"}), 400
        
        required_fields = ["image_data", "cylinder_condition"]
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Campo requerido: {field}"}), 400
        
        # Validar condición del cilindro
        valid_conditions = ["correct", "dented", "false_positive"]
        if data["cylinder_condition"] not in valid_conditions:
            return jsonify({
                "success": False, 
                "error": f"cylinder_condition debe ser uno de: {valid_conditions}"
            }), 400
        
        # Procesar imagen
        image_data = data["image_data"]
        cylinder_condition = data["cylinder_condition"]
        confidence_score = data.get("confidence_score", 1.0)
        source_info = data.get("source_info", {})
        
        # Decodificar imagen base64
        try:
            if image_data.startswith('data:'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({"success": False, "error": f"Error decodificando imagen: {e}"}), 400
        
        # Subir imagen usando Gemini
        point_id = processor.upload_image(
            image=image_bytes,
            cylinder_condition=cylinder_condition,
            confidence_score=confidence_score,
            source="n8n",
            verified=False,
            additional_metadata=source_info
        )
        
        logger.info(f"Imagen procesada con Gemini exitosamente: {point_id}")
        
        return jsonify({
            "success": True,
            "point_id": point_id,
            "message": "Imagen procesada y almacenada exitosamente con Gemini",
            "cylinder_condition": cylinder_condition,
            "confidence_score": confidence_score,
            "embedding_model": processor.embedding_model,
            "vision_model":    processor.vision_model_id,
        })
        
    except Exception as e:
        logger.error(f"Error procesando imagen: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/classify-image', methods=['POST'])
def classify_image():
    """
    Clasificar imagen usando RAG con Gemini
    
    Body JSON:
    {
        "image_data": "base64_encoded_image",
        "confidence_threshold": 0.8
    }
    """
    try:
        if processor is None:
            return jsonify({"success": False, "error": "Procesador no disponible"}), 500
        
        # Validar datos de entrada
        data = request.get_json()
        if not data or "image_data" not in data:
            return jsonify({"success": False, "error": "Campo image_data requerido"}), 400
        
        # Decodificar imagen
        image_data = data["image_data"]
        confidence_threshold = data.get("confidence_threshold", 0.8)
        
        try:
            if image_data.startswith('data:'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({"success": False, "error": f"Error decodificando imagen: {e}"}), 400
        
        # Clasificar imagen usando Gemini
        classification_result = processor.classify_cylinder(
            image=image_bytes,
            confidence_threshold=confidence_threshold
        )
        
        logger.info(f"Imagen clasificada con Gemini: {classification_result['predicted_condition']}")
        
        return jsonify({
            "success":    True,
            "classification": classification_result,
            "model_used": processor.vision_model_id,
        })
        
    except Exception as e:
        logger.error(f"Error clasificando imagen: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/search-similar', methods=['POST'])
def search_similar():
    """
    Buscar imágenes similares
    
    Body JSON:
    {
        "image_data": "base64_encoded_image",
        "limit": 5,
        "score_threshold": 0.7,
        "filter_condition": "dented"
    }
    """
    try:
        if processor is None:
            return jsonify({"success": False, "error": "Procesador no disponible"}), 500
        
        # Validar datos de entrada
        data = request.get_json()
        if not data or "image_data" not in data:
            return jsonify({"success": False, "error": "Campo image_data requerido"}), 400
        
        # Parámetros de búsqueda
        image_data = data["image_data"]
        limit = data.get("limit", 5)
        score_threshold = data.get("score_threshold", 0.5)
        filter_condition = data.get("filter_condition")
        
        try:
            if image_data.startswith('data:'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({"success": False, "error": f"Error decodificando imagen: {e}"}), 400
        
        # Buscar imágenes similares usando Gemini
        similar_images = processor.search_similar_images(
            query_image=image_bytes,
            limit=limit,
            score_threshold=score_threshold,
            filter_condition=filter_condition
        )
        
        logger.info(f"Encontradas {len(similar_images)} imágenes similares")
        
        return jsonify({
            "success": True,
            "similar_images": similar_images,
            "count": len(similar_images),
            "model_used": f"{processor.vision_model_id} + {processor.embedding_model}"
        })
        
    except Exception as e:
        logger.error(f"Error buscando imágenes similares: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Obtener estadísticas de la base de datos"""
    try:
        if processor is None:
            return jsonify({"success": False, "error": "Procesador no disponible"}), 500
        
        # Obtener información de la colección
        collection_info = processor.qdrant_client.get_collection(processor.collection_name)
        
        stats = {
            "total_images": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "distance": str(collection_info.config.params.vectors.distance),
            "indexed_vectors": collection_info.indexed_vectors_count,
            "qdrant_type": "cloud" if processor.qdrant_cloud_url else "local",
            "embedding_model": processor.embedding_model,
            "vision_model":    processor.vision_model_id,
            "sdk":             "google-genai"
        }
        
        return jsonify({
            "success": True,
            "stats": stats
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint no encontrado"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Error interno del servidor"}), 500

if __name__ == '__main__':
    # Configuración del servidor
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "5000"))
    debug = os.getenv("API_DEBUG", "False").lower() == "true"
    
    # Verificar configuración antes de iniciar
    print("=== Iniciando API Server ===")
    print(f"Host: {host}:{port}")
    print(f"Gemini API Key: {'✓ Configurada' if os.getenv('GEMINI_API_KEY') else '✗ Falta'}")
    print(f"Qdrant Cloud URL: {'✓ Configurada' if os.getenv('QDRANT_CLOUD_URL') else '✗ Falta'}")
    print(f"Qdrant API Key: {'✓ Configurada' if os.getenv('QDRANT_API_KEY') else '✗ Falta'}")
    print("=" * 40)
    
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  ADVERTENCIA: GEMINI_API_KEY no está configurada")
    if not os.getenv("QDRANT_CLOUD_URL"):
        print("⚠️  ADVERTENCIA: QDRANT_CLOUD_URL no está configurada")
    
    if processor is None:
        print("❌ ERROR: No se pudo inicializar el procesador")
        print("Verifica las variables de entorno")
        exit(1)
    
    print("✅ Servidor listo")
    print(f"\nIniciando en {host}:{port}...")
    
    app.run(host=host, port=port, debug=debug)