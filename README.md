# Sistema RAG para Clasificación de Imágenes de Cilindros

Este sistema utiliza **Google Gemini** para generar embeddings multimodales y **Qdrant Cloud** como base de datos vectorial para implementar un sistema RAG (Retrieval-Augmented Generation) especializado en la clasificación de imágenes de cilindros industriales.

## Características

- ✅ Clasificación automática de cilindros en 3 categorías usando Gemini Vision
- ✅ Integración con Qdrant Cloud (servicio en línea)
- ✅ Integración con n8n para automatización
- ✅ API REST para procesamiento de imágenes
- ✅ Búsqueda por similitud usando embeddings de Gemini
- ✅ Sistema de confianza y verificación humana
- ✅ Descripciones automáticas de imágenes

## Categorías de Clasificación

1. **correct**: Cilindro en buen estado, sin daños visibles
2. **dented**: Cilindro con abolladuras o daños evidentes  
3. **false_positive**: Cilindro que parece tener daños pero está en buen estado

## Tecnologías Utilizadas

- **Google Gemini 1.5 Flash**: Análisis de imágenes y descripciones
- **Google Gemini text-embedding-004**: Generación de embeddings vectoriales
- **Qdrant Cloud**: Base de datos vectorial (servicio en línea)
- **Flask**: API REST
- **n8n**: Automatización y workflows

## Configuración Inicial

### 1. Obtener API Keys

#### Google Gemini API Key
1. Visita [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Crea una API key gratuita
3. Copia la key

#### Qdrant Cloud
1. Regístrate en [Qdrant Cloud](https://cloud.qdrant.io/)
2. Crea un cluster gratuito
3. Obtén tu URL y API Key del cluster

### 2. Instalar Dependencias

```bash
# Clonar o descargar el proyecto
cd Adendas_Antiguas

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Configurar Variables de Entorno

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar .env con tus credenciales
```

Edita el archivo `.env`:

```env
GEMINI_API_KEY=tu_api_key_de_gemini_aqui
QDRANT_CLOUD_URL=https://tu-cluster.qdrant.io:6333
QDRANT_API_KEY=tu_api_key_de_qdrant_aqui
```

### 4. Inicializar Base de Datos

```bash
python qdrant_setup.py
```

Deberías ver:
```
=== Inicialización de Base de Datos Qdrant RAG ===
Conectando a Qdrant Cloud: https://tu-cluster.qdrant.io:6333
✅ Conexión exitosa
✅ Colección creada/verificada
🎉 Base de datos inicializada correctamente
```

### 5. Iniciar API Server

```bash
python api_server.py
```

La API estará disponible en `http://localhost:5000`

### 6. Probar Sistema

```bash
python test_system.py
```

## Estructura de Metadatos

Cada imagen almacenada contiene los siguientes metadatos:

```json
{
  "cylinder_condition": "correct|dented|false_positive",
  "condition_description": "Descripción de la condición",
  "description": "Descripción detallada generada por Gemini",
  "confidence_score": 0.95,
  "upload_timestamp": "2024-01-15T10:30:00.000Z",
  "source": "n8n|manual|api",
  "verified": true,
  "embedding_model": "Gemini text-embedding-004",
  "vision_model": "Gemini 1.5 Flash",
  "vector_size": 768
}
```

## Uso Básico

### Procesar Imagen desde Python

```python
from image_processor import CylinderImageProcessor

# Inicializar procesador (lee de .env automáticamente)
processor = CylinderImageProcessor()

# Subir imagen
point_id = processor.upload_image(
    image="path/to/cylinder.jpg",
    cylinder_condition="correct",
    confidence_score=0.95,
    source="manual"
)

# Clasificar nueva imagen
result = processor.classify_cylinder("path/to/test_cylinder.jpg")
print(f"Condición predicha: {result['predicted_condition']}")
print(f"Confianza: {result['confidence']}")
print(f"Descripción: {result['description']}")
```

### API Endpoints

#### GET /health
Verificar estado del sistema

```bash
curl http://localhost:5000/health
```

#### POST /process-image
Procesar y almacenar imagen

```bash
curl -X POST http://localhost:5000/process-image \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "base64_encoded_image",
    "cylinder_condition": "correct",
    "confidence_score": 0.95
  }'
```

#### POST /classify-image
Clasificar imagen usando RAG con Gemini

```bash
curl -X POST http://localhost:5000/classify-image \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "base64_encoded_image",
    "confidence_threshold": 0.7
  }'
```

#### POST /search-similar
Buscar imágenes similares

```bash
curl -X POST http://localhost:5000/search-similar \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "base64_encoded_image",
    "limit": 5,
    "filter_condition": "dented"
  }'
```

#### GET /stats
Obtener estadísticas

```bash
curl http://localhost:5000/stats
```

## Integración con n8n

### 1. Importar Workflow

1. Abre n8n
2. Crea un nuevo workflow
3. Importa el archivo `n8n_workflow_example.json`
4. Configura la URL del webhook
5. Activa el workflow

### 2. Enviar Imagen a n8n

```bash
curl -X POST https://your-n8n-instance/webhook/upload-cylinder-image \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "base64_encoded_image",
    "cylinder_condition": "correct",
    "confidence_score": 0.95,
    "source_ip": "192.168.1.100"
  }'
```

### 3. Respuesta Esperada

```json
{
  "status": "success",
  "message": "Imagen procesada y almacenada exitosamente",
  "data": {
    "point_id": "uuid-here",
    "cylinder_condition": "correct",
    "confidence_score": 0.95,
    "classification": {
      "predicted_condition": "correct",
      "confidence": 0.92,
      "is_confident": true,
      "description": "Cilindro azul en buen estado sin daños visibles"
    },
    "timestamp": "2024-01-15T10:30:00.000Z"
  }
}
```

## Flujo de Trabajo Recomendado

### 1. Entrenamiento Inicial

```python
from image_processor import CylinderImageProcessor

processor = CylinderImageProcessor()

# Subir imágenes de entrenamiento
training_images = [
    ("correct_001.jpg", "correct"),
    ("correct_002.jpg", "correct"),
    ("dented_001.jpg", "dented"),
    ("dented_002.jpg", "dented"),
    ("false_pos_001.jpg", "false_positive"),
]

for image_path, condition in training_images:
    processor.upload_image(
        image=image_path,
        cylinder_condition=condition,
        confidence_score=1.0,
        source="training",
        verified=True
    )
```

### 2. Clasificación en Producción

```python
# Clasificar nueva imagen
result = processor.classify_cylinder("new_cylinder.jpg")

if result["is_confident"]:
    print(f"✅ Clasificación: {result['predicted_condition']}")
    print(f"   Confianza: {result['confidence']:.2%}")
    print(f"   Descripción: {result['description']}")
else:
    print("⚠️ Requiere verificación humana")
    print(f"   Mejor predicción: {result['predicted_condition']}")
    print(f"   Confianza baja: {result['confidence']:.2%}")
```

### 3. Mejora Continua

```python
# Agregar nuevos ejemplos basados en feedback humano
if human_verification_result != predicted_result:
    processor.upload_image(
        image_path,
        cylinder_condition=human_verification_result,
        confidence_score=1.0,
        source="human_correction",
        verified=True
    )
```

## Configuración de n8n

### Variables de Entrada Requeridas

```json
{
  "image_base64": "string (required)",
  "cylinder_condition": "correct|dented|false_positive (required)",
  "confidence_score": "number (optional, default: 1.0)",
  "source_ip": "string (optional)",
  "user_agent": "string (optional)"
}
```

### Nodos del Workflow

1. **Webhook - Receive Image**: Recibe imágenes via POST
2. **Validate Input**: Valida datos de entrada
3. **Process Image with RAG**: Procesa imagen con API
4. **Check Success**: Verifica si el procesamiento fue exitoso
5. **Classify Similar Images**: Clasifica usando imágenes similares
6. **Success Response** / **Error Response**: Respuestas del proceso

## Ventajas de Usar Gemini

### Gemini Vision (gemini-1.5-flash)
- **Análisis multimodal**: Entiende imágenes y texto
- **Descripciones detalladas**: Genera descripciones ricas en contexto
- **Rápido**: Procesa imágenes en segundos
- **Económico**: Tier gratuito generoso
- **Multilingüe**: Soporta español nativamente

### Gemini Embeddings (text-embedding-004)
- **768 dimensiones**: Balance óptimo entre precisión y velocidad
- **Multilingüe**: Excelente para descripciones en español
- **Alta calidad**: Estado del arte en embeddings

## Costos y Límites

### Google Gemini (Tier Gratuito)
- 15 requests por minuto
- 1 millón de tokens por minuto
- 1500 requests por día
- Suficiente para desarrollo y pruebas

### Qdrant Cloud (Tier Gratuito)
- 1 cluster gratuito
- 1GB de almacenamiento
- Suficiente para miles de imágenes

## Solución de Problemas

### Error: "GEMINI_API_KEY no está configurada"

1. Verifica que el archivo `.env` existe
2. Verifica que la variable está correctamente escrita
3. Obtén una API key en: https://aistudio.google.com/app/apikey

### Error: "No se pudo conectar a Qdrant"

1. Verifica que `QDRANT_CLOUD_URL` sea correcta (incluye puerto)
2. Verifica que `QDRANT_API_KEY` sea válida
3. Formato típico: `https://xxx-xxx.aws.cloud.qdrant.io:6333`

### Error: "Cuota de Gemini excedida"

- Espera unos minutos (rate limit)
- O actualiza a plan de pago
- Considera implementar caché

### Error: "Modelo no encontrado"

Verifica que estás usando:
- `gemini-1.5-flash` (vision)
- `models/text-embedding-004` (embeddings)

## Monitoreo

### Verificar Estado del Sistema

```bash
# Verificar API
curl http://localhost:5000/health

# Ver estadísticas
curl http://localhost:5000/stats
```

### Logs

Los logs se muestran en la consola donde ejecutaste `api_server.py`.

## Archivos del Proyecto

- `qdrant_setup.py`: Script de inicialización de base de datos
- `image_processor.py`: Procesador principal con Gemini
- `api_server.py`: Servidor API Flask
- `n8n_workflow_example.json`: Workflow de ejemplo para n8n
- `test_system.py`: Script de pruebas
- `requirements.txt`: Dependencias del proyecto
- `.env.example`: Plantilla de variables de entorno
- `README.md`: Esta documentación

## Seguridad

⚠️ **IMPORTANTE**:
- Nunca subas el archivo `.env` a git
- Mantén tus API keys seguras
- Usa variables de entorno en producción
- Considera rotar las keys periódicamente

Agrega `.env` a tu `.gitignore`:
```bash
echo ".env" >> .gitignore
```

## Próximos Pasos

1. **Agregar más categorías**: Expandir las clasificaciones
2. **Mejorar precisión**: Ajustar umbrales y parámetros
3. **Dashboard web**: Interfaz para gestión
4. **Detección automática**: Pipeline desde cámara
5. **Alertas**: Notificaciones para defectos críticos
6. **Reportes**: Análisis estadísticos periódicos

## Soporte

Para problemas:
1. Revisa los logs de la aplicación
2. Verifica las variables de entorno
3. Consulta documentación de [Gemini](https://ai.google.dev/docs) y [Qdrant](https://qdrant.tech/documentation/)
4. Verifica que las API keys sean válidas