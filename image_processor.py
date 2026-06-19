#!/usr/bin/env python3
"""
Procesador de imágenes para base de datos RAG de cilindros
Usa google-genai SDK (nueva) + gemini-2.5-flash + gemini-embedding-2
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
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dimensión de gemini-embedding-2 (modelo multimodal Google, output por defecto)
EMBEDDING_DIM   = 3072
EMBEDDING_MODEL = "models/gemini-embedding-2"
VISION_MODEL    = "gemini-2.5-flash"


class CylinderImageProcessor:
    def __init__(self,
                 gemini_api_key: Optional[str] = None,
                 qdrant_cloud_url: Optional[str] = None,
                 qdrant_api_key: Optional[str] = None,
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333):
        """
        Inicializar el procesador de imágenes con la nueva google-genai SDK.

        Args:
            gemini_api_key:   API Key de Google Gemini
            qdrant_cloud_url: URL de Qdrant Cloud
            qdrant_api_key:   API Key de Qdrant Cloud
            qdrant_host:      Host de Qdrant local (fallback)
            qdrant_port:      Puerto de Qdrant local (fallback)
        """
        self.collection_name = "cylinder_images"
        self.vector_size      = EMBEDDING_DIM
        self.embedding_model  = EMBEDDING_MODEL
        self.vision_model_id  = VISION_MODEL

        # ── Gemini clients (nueva SDK) ────────────────────────────────────
        # IMPORTANTE: Esta API key requiere v1 para embeddings y v1beta para
        # generateContent (systemInstruction no existe en v1).
        # Por eso usamos DOS clientes separados.
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY es requerido. "
                "Obtén una en https://aistudio.google.com/app/apikey"
            )

        # Cliente v1 → usado para embedContent (embeddings)
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1")
        )
        # Cliente v1beta → usado para generateContent (vision + system_instruction)
        self.client_v1beta = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta")
        )
        logger.info(f"Clientes Gemini inicializados | visión: {VISION_MODEL} | "
                    f"embeddings: {EMBEDDING_MODEL} ({EMBEDDING_DIM}d) | "
                    f"SDK: google-genai v2 | v1 (embed) + v1beta (vision)")

        # ── Qdrant (priorizar Cloud) ───────────────────────────────────────
        self.qdrant_cloud_url = qdrant_cloud_url or os.getenv("QDRANT_CLOUD_URL")
        self.qdrant_api_key   = qdrant_api_key   or os.getenv("QDRANT_API_KEY")
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
            "correct":        "Cilindro en buen estado, sin daños visibles",
            "dented":         "Cilindro con abolladuras o daños evidentes",
            "false_positive": "Cilindro que parece tener daños pero está en buen estado",
        }

    # ── Utilidades de imagen ──────────────────────────────────────────────

    def _to_pil(self, image: Union[str, Image.Image, bytes]) -> Image.Image:
        """Normalizar cualquier entrada a PIL Image RGB."""
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        raise ValueError(f"Tipo de imagen no soportado: {type(image)}")

    def _pil_to_bytes(self, pil_image: Image.Image, fmt: str = "JPEG") -> bytes:
        """Convertir PIL Image a bytes."""
        buf = io.BytesIO()
        pil_image.save(buf, format=fmt)
        return buf.getvalue()

    # ── Prompts por defecto ───────────────────────────────────────────────
    DEFAULT_SYSTEM_INSTRUCTION = (
        "Eres un inspector experto de cilindros industriales. "
        "Analiza la imagen con foco en detectar abolladuras, golpes, "
        "deformaciones y cualquier daño visible en la superficie. "
        "Sé estricto: si dudas si hay daño, indícalo explícitamente."
    )

    # System prompt detallado de Crosland (para máxima precisión en
    # control de calidad e integridad estructural de cilindros).
    CROSLAND_SYSTEM_INSTRUCTION = (
        "Eres un inspector experto en control de calidad e integridad "
        "estructural de cilindros de alta presión y componentes industriales "
        "de Crosland. Tu tarea es analizar imágenes de superficies metálicas "
        "y reportar anomalías reales, diferenciando estrictamente los defectos "
        "de manipulación de las marcas de fabricación.\n\n"
        "Aplica los siguientes criterios de exclusión visual con precisión "
        "quirúrgica:\n\n"
        "1. DIFERENCIACIÓN DE ESTAMPADO VS. PICADURAS DE MANIPULACIÓN:\n"
        "   - Estampado (Marcas Intencionales): Se caracteriza por caracteres "
        "alfanuméricos con bordes definidos, geometría uniforme, profundidad "
        "constante y un patrón alineado de presión mecánica (prensado). Si notas "
        "irregularidades justo en los bordes de los números (como el '8' o el '6'), "
        "analiza si la deformación es interna al carácter o externa.\n"
        "   - Picaduras por Manipulación/Impacto (Defectos): Son depresiones "
        "aleatorias, asimétricas, con fondos rugosos o puntiagudos que rompen la "
        "continuidad del material de forma caótica. No siguen la geometría de "
        "ninguna tipografía. Si un impacto ocurre 'cerca de' o 'sobre' un número "
        "estampado, clasifícalo como 'Daño por impacto/manipulación posterior' y "
        "NO como un defecto del estampado.\n\n"
        "2. TEXTURA Y ACABADO SUPERFICIAL:\n"
        "   - Identifica zonas con pintura descascarada, rugosidad localizada o "
        "cambios de tonalidad (gris oscuro/oxidación) alrededor de los estampados. "
        "Si la irregularidad carece de bordes rectos o simétricos, descarta "
        "problemas de estampado y repórtalo como 'Desgaste por fricción, picaduras "
        "por manipulación mecánica o corrosión focalizada'.\n\n"
        "3. PROTOCOLO DE CONCORDANCIA CON CONTEXTO (QDRANT):\n"
        "   - Al procesar la consulta, recibirás un contexto de imágenes similares "
        "recuperadas desde Qdrant. Utiliza estos vectores de referencia para "
        "comparar visualmente si el cilindro analizado presenta el estándar de "
        "fábrica aprobado o si los patrones de sombra y profundidad en la zona de "
        "caracteres coinciden con cilindros previamente clasificados como 'dañados "
        "por golpe/picadura'.\n\n"
        "Sé técnico, directo, descriptivo y evita generar falsos positivos "
        "basados únicamente en la presencia de texto estampado legítimo."
    )

    DEFAULT_USER_PROMPT = (
        "Analiza esta imagen de un cilindro industrial en detalle. "
        "Describe específicamente:\n"
        "1. El estado general del cilindro\n"
        "2. Si tiene abolladuras, golpes o deformaciones visibles "
        "(indica su ubicación aproximada y tamaño)\n"
        "3. Si la superficie parece dañada o en buen estado\n"
        "4. Color, forma y características visibles\n"
        "5. Cualquier detalle relevante sobre su condición\n\n"
        "Responde en español de forma concisa y técnica."
    )

    # Temperatura baja para análisis más determinístico
    DEFAULT_TEMPERATURE = 0.2

    # ── Gemini Vision ─────────────────────────────────────────────────────

    def get_image_description(
        self,
        image: Union[str, Image.Image, bytes],
        system_instruction: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generar descripción detallada de la imagen con gemini-2.5-flash.

        Args:
            image:               Imagen a analizar
            system_instruction:  Rol/instrucción de sistema (opcional).
                                 Personaliza el comportamiento del modelo.
                                 Si es None, usa DEFAULT_SYSTEM_INSTRUCTION.
            user_prompt:         Prompt del usuario (opcional).
                                 Si es None, usa DEFAULT_USER_PROMPT.
            temperature:         Temperatura de generación (default 0.2).
                                 Bajo = más determinístico.

        Returns:
            str: Descripción generada por Gemini
        """
        try:
            pil_image = self._to_pil(image)
            img_bytes  = self._pil_to_bytes(pil_image)

            sys_instr = system_instruction or self.DEFAULT_SYSTEM_INSTRUCTION
            usr_prmpt = user_prompt or self.DEFAULT_USER_PROMPT
            temp      = temperature if temperature is not None else self.DEFAULT_TEMPERATURE

            # generateContent requiere v1beta para soportar systemInstruction
            response = self.client_v1beta.models.generate_content(
                model=self.vision_model_id,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    types.Part.from_text(text=usr_prmpt),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=sys_instr,
                    temperature=temp,
                ),
            )

            description = response.text.strip()
            logger.debug(f"Descripción generada: {description[:120]}…")
            return description

        except Exception as e:
            logger.error(f"Error generando descripción: {e}")
            return f"Error generando descripción: {e}"

    # ── Gemini Embeddings ─────────────────────────────────────────────────

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generar embedding de 3072 dimensiones con gemini-embedding-2.

        Usa client.models.embed_content() según la nueva SDK.
        """
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config=types.EmbedContentConfig(task_type="retrieval_document"),
            )
            # result.embeddings es una lista de ContentEmbedding
            embedding = np.array(result.embeddings[0].values, dtype=np.float32)

            if embedding.shape[0] != EMBEDDING_DIM:
                logger.warning(
                    f"Dimensión inesperada del embedding: "
                    f"{embedding.shape[0]} (esperado {EMBEDDING_DIM})"
                )
            return embedding

        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            raise

    # ── Pipeline RAG ──────────────────────────────────────────────────────

    def process_image_for_rag(
        self,
        image: Union[str, Image.Image, bytes],
        system_instruction: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Pipeline completo: imagen → descripción (Gemini Vision) → embedding.

        Args:
            image:              Imagen a procesar
            system_instruction: System instruction opcional para Gemini
            user_prompt:        Prompt de usuario opcional para Gemini
            temperature:        Temperatura de Gemini (default 0.2)

        Returns:
            (embedding np.ndarray 3072d, descripción str)
        """
        logger.info("Paso 1/2 → Gemini Vision: generando descripción...")
        description = self.get_image_description(
            image, system_instruction=system_instruction,
            user_prompt=user_prompt, temperature=temperature,
        )

        logger.info("Paso 2/2 → Gemini Embeddings: vectorizando descripción...")
        embedding = self.generate_embedding(description)

        logger.info(f"Pipeline RAG completado | dim={embedding.shape[0]}")
        return embedding, description

    def upload_image(
        self,
        image: Union[str, Image.Image, bytes],
        cylinder_condition: str,
        confidence_score: float = 1.0,
        source: str = "manual",
        verified: bool = True,
        additional_metadata: Optional[Dict] = None,
        system_instruction: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> str:
        """
        Procesar imagen y almacenarla en Qdrant Cloud.

        Args:
            image:              Imagen a subir
            cylinder_condition: 'correct' | 'dented' | 'false_positive'
            confidence_score:   Confianza de la etiqueta (0.0 – 1.0)
            source:             Origen del dato (n8n, manual, training…)
            verified:           Si fue verificado por un humano
            additional_metadata: Metadatos extra a guardar
            system_instruction: System instruction opcional para Gemini
            user_prompt:        Prompt de usuario opcional para Gemini
            temperature:        Temperatura de Gemini (default 0.2)
            reason:             Razón de la clasificación (ej: 'Estampado '
                                'confundido con abolladura, es falso positivo').
                                Importante para entrenar el RAG en falsos
                                positivos/negativos.

        Returns:
            str: UUID del punto insertado en Qdrant
        """
        if cylinder_condition not in self.categories:
            raise ValueError(
                f"Categoría inválida '{cylinder_condition}'. "
                f"Opciones: {list(self.categories.keys())}"
            )

        embedding, description = self.process_image_for_rag(
            image, system_instruction=system_instruction,
            user_prompt=user_prompt, temperature=temperature,
        )

        point_id = str(uuid.uuid4())

        payload = {
            "cylinder_condition":    cylinder_condition,
            "condition_description": self.categories[cylinder_condition],
            "description":           description,
            "confidence_score":      float(confidence_score),
            "upload_timestamp":      datetime.now().isoformat(),
            "source":                source,
            "verified":              verified,
            "embedding_model":       EMBEDDING_MODEL,
            "vision_model":          VISION_MODEL,
            "vector_size":           int(embedding.shape[0]),
        }

        # Guardar razón como metadata para que el RAG la use en futuras
        # clasificaciones (clave para falsos positivos / falsos negativos)
        if reason:
            payload["reason"] = reason.strip()

        if additional_metadata:
            payload.update(additional_metadata)

        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=embedding.tolist(), payload=payload)],
        )

        logger.info(f"Imagen almacenada | id={point_id} | condición={cylinder_condition}")
        return point_id

    def search_similar_images(
        self,
        query_image: Union[str, Image.Image, bytes],
        limit: int = 5,
        score_threshold: float = 0.5,
        filter_condition: Optional[str] = None,
        system_instruction: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> List[Dict]:
        """
        Buscar imágenes similares por similitud coseno en Qdrant.

        Args:
            query_image:        Imagen de consulta
            limit:              Máximo de resultados
            score_threshold:    Similitud mínima
            filter_condition:   Filtrar por 'correct' | 'dented' | 'false_positive'
            system_instruction: System instruction opcional para Gemini
            user_prompt:        Prompt de usuario opcional para Gemini
            temperature:        Temperatura de Gemini (default 0.2)

        Returns:
            Lista de dicts con {id, score, metadata}
        """
        query_embedding, _ = self.process_image_for_rag(
            query_image, system_instruction=system_instruction,
            user_prompt=user_prompt, temperature=temperature,
        )

        query_filter = None
        if filter_condition:
            query_filter = {
                "must": [{"key": "cylinder_condition", "match": {"value": filter_condition}}]
            }

        # qdrant-client >= 1.10 usa query_points() en lugar de search()
        response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
        )
        hits = response.points

        results = [{"id": h.id, "score": h.score, "metadata": h.payload} for h in hits]
        logger.info(f"Búsqueda completada: {len(results)} resultados")
        return results

    def classify_cylinder(
        self,
        image: Union[str, Image.Image, bytes],
        confidence_threshold: float = 0.8,
        system_instruction: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict:
        """
        Clasificar estado del cilindro mediante RAG (votación ponderada).

        El contexto recuperado de Qdrant se envía de vuelta a gemini-2.5-flash
        para que genere una respuesta enriquecida final. Si las imágenes
        similares tienen 'reason' guardada, se incluye en el contexto para
        que Gemini aprenda de falsos positivos/negativos previos.

        Args:
            image:                Imagen a clasificar
            confidence_threshold: Umbral mínimo para considerar la predicción confiable
            system_instruction:   System instruction opcional para Gemini
            user_prompt:          Prompt de usuario opcional para Gemini
            temperature:          Temperatura de Gemini (default 0.2)

        Returns:
            Dict con predicted_condition, confidence, description, rag_explanation, …
        """
        # ── 1. Describir imagen de entrada ────────────────────────────────
        description = self.get_image_description(
            image, system_instruction=system_instruction,
            user_prompt=user_prompt, temperature=temperature,
        )

        # ── 2. Recuperar contexto de Qdrant ───────────────────────────────
        similar_images = self.search_similar_images(
            query_image=image, limit=10, score_threshold=0.4,
            system_instruction=system_instruction, user_prompt=user_prompt,
            temperature=temperature,
        )

        if not similar_images:
            return {
                "predicted_condition": "unknown",
                "confidence":          0.0,
                "is_confident":        False,
                "description":         description,
                "rag_explanation":     "No hay imágenes de referencia en la base de datos.",
                "similar_images_count": 0,
            }

        # ── 3. Votación ponderada ─────────────────────────────────────────
        votes: Dict[str, float] = {"correct": 0.0, "dented": 0.0, "false_positive": 0.0}
        total = 0.0
        for hit in similar_images:
            cond   = hit["metadata"]["cylinder_condition"]
            weight = hit["score"] * hit["metadata"].get("confidence_score", 1.0)
            votes[cond] += weight
            total        += weight

        predicted  = max(votes, key=votes.get)
        confidence = votes[predicted] / total if total > 0 else 0.0

        # ── 4. Generar explicación enriquecida con gemini-2.5-flash ───────
        # Las imágenes similares pueden traer 'reason' (falso positivo/negativo);
        # esto le enseña a Gemini a NO repetir errores pasados.
        def _format_context_line(h):
            cond = h["metadata"]["cylinder_condition"]
            desc = h["metadata"].get("description", "")[:120]
            reason = h["metadata"].get("reason", "")
            line = f"- Similitud {h['score']:.2f}: {cond} | {desc}"
            if reason:
                line += f"\n    ⚠ Razón registrada: {reason}"
            return line

        context_lines = "\n".join(_format_context_line(h) for h in similar_images[:5])

        # Usar el system_instruction del usuario si existe; si no, el default
        sys_instr = system_instruction or self.DEFAULT_SYSTEM_INSTRUCTION
        temp      = temperature if temperature is not None else self.DEFAULT_TEMPERATURE

        rag_prompt = (
            f"Descripción de la imagen analizada:\n{description}\n\n"
            f"Ejemplos similares encontrados en la base de datos:\n{context_lines}\n\n"
            f"Clasificación automática: '{predicted}' (confianza {confidence:.0%}).\n\n"
            f"Si alguno de los ejemplos tiene 'Razón registrada', úsala como pista: "
            f"si la imagen se parece a un falso positivo previo, NO la marques como "
            f"dañada; si se parece a un falso negativo, SÍ marca el daño.\n\n"
            f"Proporciona una breve explicación técnica en español de por qué esta imagen "
            f"corresponde a la categoría '{predicted}', basándote en los ejemplos anteriores."
        )

        try:
            # generateContent con systemInstruction requiere v1beta
            rag_response = self.client_v1beta.models.generate_content(
                model=self.vision_model_id,
                contents=rag_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instr,
                    temperature=temp,
                ),
            )
            rag_explanation = rag_response.text.strip()
        except Exception as e:
            logger.warning(f"No se pudo generar explicación RAG: {e}")
            rag_explanation = f"Clasificado como '{predicted}' por votación ponderada."

        return {
            "predicted_condition":  predicted,
            "confidence":           float(confidence),
            "is_confident":         confidence >= confidence_threshold,
            "description":          description,
            "rag_explanation":      rag_explanation,
            "condition_scores":     votes,
            "similar_images_count": len(similar_images),
            "best_match_score":     similar_images[0]["score"],
        }


# ── Helpers para n8n ──────────────────────────────────────────────────────────

def process_image_from_n8n(
    image_data: str,
    cylinder_condition: str,
    confidence_score: float = 1.0,
) -> Dict:
    """
    Punto de entrada simplificado para n8n.

    Args:
        image_data:         Imagen en base64 (con o sin prefijo data:…)
        cylinder_condition: 'correct' | 'dented' | 'false_positive'
        confidence_score:   Confianza de la etiqueta

    Returns:
        Dict con success, point_id / error
    """
    try:
        if image_data.startswith("data:"):
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        processor = CylinderImageProcessor()
        point_id  = processor.upload_image(
            image=image_bytes,
            cylinder_condition=cylinder_condition,
            confidence_score=confidence_score,
            source="n8n",
            verified=False,
        )
        return {"success": True, "point_id": point_id, "message": "Imagen procesada con Gemini"}

    except Exception as e:
        return {"success": False, "error": str(e), "message": "Error procesando imagen"}


if __name__ == "__main__":
    try:
        proc = CylinderImageProcessor()
        print("=== Procesador de Imágenes de Cilindros ===")
        print(f"SDK:             google-genai (nueva)")
        print(f"Modelo visión:   {proc.vision_model_id}")
        print(f"Modelo embedding:{proc.embedding_model}")
        print(f"Dimensión vector:{proc.vector_size}")
        print(f"Qdrant:          {'Cloud' if proc.qdrant_cloud_url else 'Local'}")
    except Exception as e:
        print(f"Error inicializando procesador: {e}")
