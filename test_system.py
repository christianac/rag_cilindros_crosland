#!/usr/bin/env python3
"""
Script de prueba para el sistema RAG con Gemini y Qdrant Cloud
"""

import base64
import requests
import json
import time
from PIL import Image, ImageDraw
import io
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def create_test_image(color='red', size=(224, 224), text='CYLINDER'):
    """Crear imagen de prueba simple con texto"""
    img = Image.new('RGB', size, color=color)
    draw = ImageDraw.Draw(img)
    
    # Agregar texto
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except:
        font = None
    
    # Calcular posición del texto
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        text_width = len(text) * 6
        text_height = 11
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Dibujar texto
    draw.text((x, y), text, fill='white', font=font)
    
    # Agregar una "abolladura" simulada para cilindros dented
    if 'dented' in text.lower():
        draw.ellipse([50, 50, 150, 150], fill='darkblue', outline='black', width=3)
    
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode()

def check_environment():
    """Verificar que las variables de entorno estén configuradas"""
    print("=== Verificación de Configuración ===\n")
    
    required_vars = {
        "GEMINI_API_KEY": "Google Gemini API",
        "QDRANT_CLOUD_URL": "Qdrant Cloud URL",
        "QDRANT_API_KEY": "Qdrant API Key"
    }
    
    all_ok = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mostrar solo los primeros caracteres para seguridad
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"✅ {description}: {masked}")
        else:
            print(f"❌ {description}: NO CONFIGURADA")
            all_ok = False
    
    print()
    return all_ok

def test_api_endpoints():
    """Probar todos los endpoints de la API"""
    base_url = os.getenv("API_URL", "http://localhost:5000")
    
    print("=== Prueba del Sistema RAG con Gemini y Qdrant Cloud ===\n")
    
    # 1. Verificar salud del API
    print("1. Verificando salud del API...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ API funcionando correctamente")
            data = response.json()
            print(f"   Estado: {data['status']}")
            if 'config' in data:
                config = data['config']
                print(f"   Qdrant: {config.get('qdrant_type', 'unknown')}")
                print(f"   Modelo de visión: {config.get('vision_model', 'unknown')}")
                print(f"   Modelo de embeddings: {config.get('embedding_model', 'unknown')}")
        else:
            print(f"❌ API no responde correctamente: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error conectando a API: {e}")
        return False
    
    print()
    
    # 2. Subir imagen de prueba
    print("2. Subiendo imagen de cilindro CORRECTO...")
    test_image = create_test_image('blue', (224, 224), 'CORRECT')
    
    process_data = {
        "image_data": test_image,
        "cylinder_condition": "correct",
        "confidence_score": 0.95,
        "source_info": {
            "test": True,
            "description": "Imagen de prueba - cilindro correcto"
        }
    }
    
    try:
        response = requests.post(
            f"{base_url}/process-image",
            json=process_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Imagen procesada exitosamente con Gemini")
            print(f"   Point ID: {result['point_id']}")
            print(f"   Modelo: {result.get('embedding_model', 'unknown')}")
        else:
            print(f"❌ Error procesando imagen: {response.json()}")
            return False
    except Exception as e:
        print(f"❌ Error en request: {e}")
        return False
    
    print()
    
    # 3. Subir imagen de cilindro con abolladura
    print("3. Subiendo imagen de cilindro DENTED...")
    test_image_dented = create_test_image('red', (224, 224), 'DENTED')
    
    process_data = {
        "image_data": test_image_dented,
        "cylinder_condition": "dented",
        "confidence_score": 0.95,
        "source_info": {
            "test": True,
            "description": "Imagen de prueba - cilindro con abolladuras"
        }
    }
    
    try:
        response = requests.post(
            f"{base_url}/process-image",
            json=process_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Imagen con abolladuras procesada exitosamente")
        else:
            print(f"❌ Error: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # 4. Probar clasificación
    print("4. Probando clasificación de imagen con Gemini...")
    test_image_classify = create_test_image('green', (224, 224), 'TEST')
    
    classify_data = {
        "image_data": test_image_classify,
        "confidence_threshold": 0.3
    }
    
    try:
        response = requests.post(
            f"{base_url}/classify-image",
            json=classify_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Clasificación completada con Gemini")
            classification = result['classification']
            print(f"   Condición predicha: {classification['predicted_condition']}")
            print(f"   Confianza: {classification['confidence']:.2f}")
            print(f"   Es confiable: {classification['is_confident']}")
            print(f"   Imágenes similares encontradas: {classification['similar_images_count']}")
            if classification.get('description'):
                desc = classification['description'][:100]
                print(f"   Descripción: {desc}...")
        else:
            print(f"❌ Error clasificando: {response.json()}")
    except Exception as e:
        print(f"❌ Error en clasificación: {e}")
    
    print()
    
    # 5. Buscar imágenes similares
    print("5. Buscando imágenes similares...")
    
    search_data = {
        "image_data": test_image,
        "limit": 3,
        "score_threshold": 0.1
    }
    
    try:
        response = requests.post(
            f"{base_url}/search-similar",
            json=search_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Búsqueda completada con Gemini")
            print(f"   Imágenes encontradas: {result['count']}")
            
            for i, img in enumerate(result['similar_images'][:3]):
                print(f"   Resultado {i+1}:")
                print(f"     Score: {img['score']:.3f}")
                print(f"     Condición: {img['metadata']['cylinder_condition']}")
                if 'description' in img['metadata']:
                    desc = img['metadata']['description'][:80]
                    print(f"     Descripción: {desc}...")
        else:
            print(f"❌ Error buscando: {response.json()}")
    except Exception as e:
        print(f"❌ Error en búsqueda: {e}")
    
    print()
    
    # 6. Obtener estadísticas
    print("6. Obteniendo estadísticas...")
    
    try:
        response = requests.get(f"{base_url}/stats")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Estadísticas obtenidas")
            stats = result['stats']
            print(f"   Total de imágenes: {stats['total_images']}")
            print(f"   Tamaño de vector: {stats['vector_size']}")
            print(f"   Tipo de Qdrant: {stats.get('qdrant_type', 'unknown')}")
            print(f"   Modelo de embeddings: {stats.get('embedding_model', 'unknown')}")
        else:
            print(f"❌ Error obteniendo estadísticas: {response.json()}")
    except Exception as e:
        print(f"❌ Error obteniendo stats: {e}")
    
    print()
    print("=== Pruebas completadas ===")
    return True

def test_n8n_format():
    """Probar formato compatible con n8n"""
    print("\n=== Formato para n8n ===\n")
    
    # Simular datos que vendrían de n8n
    n8n_data = {
        "image_base64": create_test_image('yellow', (300, 300), 'N8N_TEST'),
        "cylinder_condition": "dented",
        "confidence_score": 0.87,
        "source_ip": "192.168.1.100",
        "user_agent": "n8n-webhook"
    }
    
    print("Datos de ejemplo para n8n webhook:")
    example_json = {k: v if k != 'image_base64' else '[base64_image_data]' 
                   for k, v in n8n_data.items()}
    print(json.dumps(example_json, indent=2))
    
    print("\nEjemplo de curl para n8n webhook:")
    print("""
curl -X POST http://your-n8n-instance/webhook/upload-cylinder-image \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_base64": "[base64_encoded_image]",
    "cylinder_condition": "correct",
    "confidence_score": 0.95,
    "source_ip": "192.168.1.100"
  }'
    """)

def main():
    """Función principal"""
    print("Iniciando pruebas del sistema...")
    print()
    
    # Verificar variables de entorno
    if not check_environment():
        print("\n❌ Faltan variables de entorno necesarias")
        print("\nPara configurar:")
        print("1. Copia .env.example a .env")
        print("2. Obtén tu API key de Gemini: https://aistudio.google.com/app/apikey")
        print("3. Crea una cuenta en Qdrant Cloud: https://cloud.qdrant.io/")
        print("4. Configura las variables en .env")
        return
    
    time.sleep(1)
    
    # Probar endpoints
    success = test_api_endpoints()
    
    if success:
        # Mostrar información sobre n8n
        test_n8n_format()
        
        print("\n🎉 Todas las pruebas completadas exitosamente!")
        print("\nPróximos pasos:")
        print("1. Importar workflow n8n desde n8n_workflow_example.json")
        print("2. Configurar webhook URL en n8n")
        print("3. Comenzar a subir imágenes reales de cilindros")
        print("4. Monitorear clasificaciones y ajustar si es necesario")
    else:
        print("\n❌ Algunas pruebas fallaron")
        print("\nVerifica:")
        print("1. El archivo .env está configurado correctamente")
        print("2. GEMINI_API_KEY es válida")
        print("3. QDRANT_CLOUD_URL y QDRANT_API_KEY son correctas")
        print("4. La API server está ejecutándose (python api_server.py)")

if __name__ == "__main__":
    main()