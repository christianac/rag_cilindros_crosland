#!/usr/bin/env python3
"""
Procesador de imágenes para base de datos RAG de cilindros
Usa google-genai SDK (nueva) + gemini-2.5-flash + gemini-embedding-2
"""

import os
import io
import json
import base64
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any
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

    # System prompt de Crosland — clasificación binaria estricta.
    # Solo dos categorías: 'correct' (válido) o 'dented' (manipulado).
    # 'false_positive' se usa solo como metadata de entrenamiento.
    CROSLAND_SYSTEM_INSTRUCTION = (
        "Rol: Actúas como un sistema experto de control de calidad industrial "
        "con tolerancia cero para alteraciones físicas. Tu tarea es clasificar "
        "la imagen de un serial metálico exclusivamente en una de dos "
        "categorías: correct o dented.\n\n"
        "Criterios estrictos de clasificación:\n\n"
        "Clasifica como dented (o incorrecto) si observas:\n\n"
        "Deformación física o abolladura por impacto: Hundimientos, muescas "
        "profundas o deformaciones mecánicas directas dentro o sobre los bordes "
        "de los caracteres grabados en relieve (Presta especial atención a "
        "golpes que distorsionen la geometría de números redondos como el 8 "
        "o el 9).\n\n"
        "Fisuras estructurales: Grietas o líneas de fractura lineales y "
        "diagonales que corten el metal o traspasen las marcas de troquelado.\n\n"
        "Clasifica como correct (o correcto) si observas únicamente:\n\n"
        "Desgaste superficial: Descascaramiento, desprendimiento de pintura "
        "(peeling), óxido superficial o rugosidades de manufactura, SIEMPRE Y "
        "CUANDO la forma geométrica y el relieve original de los caracteres "
        "troquelados no presenten hundimientos por golpes directos. La pintura "
        "saltada expone imperfecciones, pero no altera la estructura del "
        "troquel."
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

    # Reglas de OCR que se inyectan en TODOS los prompts para que Gemini
    # distinga correctamente letras vs números en códigos seriales.
    OCR_RULES = (
        "\n\nREGLAS DE LECTURA DE CARACTERES (OCR):\n"
        "- 'D' es LETRA, NUNCA número. Identifícala por su lado plano vertical derecho.\n"
        "- '0' es NÚMERO. Puede tener diagonal interna o forma ovalada.\n"
        "- '1' es NÚMERO con base. 'I' es LETRA sin base ni remate.\n"
        "- '8' es NÚMERO simétrico vertical. 'B' es LETRA asimétrica (arriba chica, abajo grande).\n"
        "- '5' es NÚMERO con ángulos rectos en base. 'S' es LETRA curva.\n"
        "- '2' es NÚMERO con base horizontal. 'Z' es LETRA con esquinas filosas.\n"
        "- 'O' es LETRA ovalada. '0' es NÚMERO (más cuadrado en tipografía industrial).\n"
        "- 'Q' es LETRA con cola inferior. '0' nunca tiene cola.\n"
        "- 'G' es LETRA con gancho interno. '6' es NÚMERO con círculo cerrado abajo.\n\n"
        "Si dudas entre letra y número en un carácter crítico, busca imágenes "
        "de referencia en el contexto RAG para confirmar."
    )

    # Temperatura baja para análisis más determinístico
    DEFAULT_TEMPERATURE = 0.2

    # Tipos de entrenamiento permitidos al subir imágenes
    TRAINING_TYPES = ("cylinder", "character")

    # ── Gemini Vision ─────────────────────────────────────────────────────

    def get_image_description(
        self,
        image: Union[str, Image.Image, bytes],
        system_instruction: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        extract_code: bool = True,
    ) -> Dict[str, Any]:
        """
        Generar descripción detallada + OCR del código del cilindro.

        Args:
            image:               Imagen a analizar
            system_instruction:  Rol/instrucción de sistema (opcional)
            user_prompt:         Prompt del usuario (opcional)
            temperature:         Temperatura de generación (default 0.2)
            extract_code:        Si True, pide JSON con OCR estructurado.
                                 Si False, devuelve solo texto libre.

        Returns:
            dict con:
                - description:     str (texto descriptivo)
                - extracted_code:  str (código OCR, vacío si no se detecta)
                - confidence_ocr:  float (confianza del OCR 0.0-1.0)
                - raw:             str (respuesta cruda por si falla parsing)
        """
        try:
            pil_image = self._to_pil(image)
            img_bytes  = self._pil_to_bytes(pil_image)

            sys_instr = system_instruction or self.DEFAULT_SYSTEM_INSTRUCTION
            usr_prmpt = user_prompt or self.DEFAULT_USER_PROMPT
            temp      = temperature if temperature is not None else self.DEFAULT_TEMPERATURE

            # ── Modo OCR: pedir JSON estructurado ────────────────────────
            if extract_code:
                ocr_user_prompt = (
                    usr_prmpt
                    + self.OCR_RULES
                    + "\n\nFORMATO DE RESPUESTA OBLIGATORIO (JSON estricto):\n"
                    + "{\n"
                    + '  "description": "<descripción técnica de la imagen en español>",\n'
                    + '  "extracted_code": "<código serial completo tal como aparece en el cilindro, sin espacios>",\n'
                    + '  "confidence_ocr": <número entre 0.0 y 1.0>\n'
                    + "}\n\n"
                    + "Reglas para 'extracted_code':\n"
                    + "- Lee cada carácter con atención (distingue D de 0, I de 1, etc.)\n"
                    + "- Si NO puedes leer ningún código, devuelve string vacío ''\n"
                    + "- Mantén el orden EXACTO en que aparecen los caracteres\n"
                    + "- NO agregues separadores que no estén en el original"
                )
                response = self.client_v1beta.models.generate_content(
                    model=self.vision_model_id,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=ocr_user_prompt),
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=sys_instr,
                        temperature=temp,
                        response_mime_type="application/json",
                    ),
                )

                raw = response.text.strip()
                try:
                    data = json.loads(raw)
                    return {
                        "description":    str(data.get("description", "")).strip(),
                        "extracted_code": str(data.get("extracted_code", "")).strip(),
                        "confidence_ocr": float(data.get("confidence_ocr", 0.0)),
                        "raw":            raw,
                    }
                except Exception as parse_err:
                    logger.warning(f"No se pudo parsear JSON, fallback a texto plano: {parse_err}")
                    return {
                        "description":    raw,
                        "extracted_code": "",
                        "confidence_ocr": 0.0,
                        "raw":            raw,
                    }

            # ── Modo texto libre ──────────────────────────────────────────
            response = self.client_v1beta.models.generate_content(
                model=self.vision_model_id,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    types.Part.from_text(text=usr_prmpt + self.OCR_RULES),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=sys_instr,
                    temperature=temp,
                ),
            )
            text = response.text.strip()
            return {
                "description":    text,
                "extracted_code": "",
                "confidence_ocr": 0.0,
                "raw":            text,
            }

        except Exception as e:
            logger.error(f"Error generando descripción: {e}")
            return {
                "description":    f"Error generando descripción: {e}",
                "extracted_code": "",
                "confidence_ocr": 0.0,
                "raw":            str(e),
            }

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
        extract_code: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Pipeline completo: imagen → (descripción + OCR) → embedding.

        Args:
            image:              Imagen a procesar
            system_instruction: System instruction opcional para Gemini
            user_prompt:        Prompt de usuario opcional para Gemini
            temperature:        Temperatura de Gemini (default 0.2)
            extract_code:       Si True, también extrae el código OCR

        Returns:
            (embedding np.ndarray 3072d, info_dict)
            info_dict = {description, extracted_code, confidence_ocr, raw}
        """
        logger.info("Paso 1/2 → Gemini Vision: descripción + OCR...")
        info = self.get_image_description(
            image, system_instruction=system_instruction,
            user_prompt=user_prompt, temperature=temperature,
            extract_code=extract_code,
        )

        logger.info("Paso 2/2 → Gemini Embeddings: vectorizando descripción...")
        # El embedding se genera SOLO sobre la descripción (no el código)
        # porque el código tiene campo dedicado y no debe contaminar el
        # espacio vectorial con texto estructurado.
        embedding = self.generate_embedding(info["description"])

        logger.info(
            f"Pipeline RAG completado | dim={embedding.shape[0]} | "
            f"código='{info['extracted_code']}'"
        )
        return embedding, info

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
        expected_code: Optional[str] = None,
        training_type: str = "cylinder",
    ) -> Dict[str, Any]:
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
            reason:             Razón de la clasificación
            expected_code:      Código serial esperado (validación humana).
                                Para training_type='character' indica qué
                                carácter está entrenando (ej: 'D', '0').
            training_type:      'cylinder' = cilindro completo (default).
                                'character' = imagen recortada de UN carácter
                                individual para entrenar OCR.

        Returns:
            dict con:
                - point_id:       str (UUID del punto insertado)
                - extracted_code: str (código OCR detectado)
                - code_match:     bool/None (True si coincide con expected_code)
        """
        if cylinder_condition not in self.categories:
            raise ValueError(
                f"Categoría inválida '{cylinder_condition}'. "
                f"Opciones: {list(self.categories.keys())}"
            )
        if training_type not in self.TRAINING_TYPES:
            raise ValueError(
                f"training_type inválido '{training_type}'. "
                f"Opciones: {self.TRAINING_TYPES}"
            )

        embedding, info = self.process_image_for_rag(
            image, system_instruction=system_instruction,
            user_prompt=user_prompt, temperature=temperature,
        )

        point_id = str(uuid.uuid4())

        payload = {
            "cylinder_condition":    cylinder_condition,
            "condition_description": self.categories[cylinder_condition],
            "description":           info["description"],
            "extracted_code":        info["extracted_code"],
            "confidence_ocr":        info["confidence_ocr"],
            "confidence_score":      float(confidence_score),
            "upload_timestamp":      datetime.now().isoformat(),
            "source":                source,
            "verified":              verified,
            "training_type":         training_type,
            "embedding_model":       EMBEDDING_MODEL,
            "vision_model":          VISION_MODEL,
            "vector_size":           int(embedding.shape[0]),
        }

        # Si el humano proporcionó el código esperado, validar match
        code_match = None
        if expected_code:
            payload["expected_code"] = expected_code.strip()
            code_match = (
                payload["extracted_code"].upper() == expected_code.strip().upper()
            )
            payload["code_match"] = code_match

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

        logger.info(
            f"Imagen almacenada | id={point_id} | tipo={training_type} | "
            f"condición={cylinder_condition} | código OCR='{info['extracted_code']}'"
        )
        return {
            "point_id":       point_id,
            "extracted_code": info["extracted_code"],
            "code_match":     code_match,
        }

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
        # ── 1. Describir imagen + extraer código OCR ──────────────────────
        info = self.get_image_description(
            image, system_instruction=system_instruction,
            user_prompt=user_prompt, temperature=temperature,
            extract_code=True,
        )
        description    = info["description"]
        extracted_code = info["extracted_code"]
        confidence_ocr = info["confidence_ocr"]

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
                "extracted_code":      extracted_code,
                "confidence_ocr":      confidence_ocr,
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
            code = h["metadata"].get("extracted_code", "")
            reason = h["metadata"].get("reason", "")
            line = f"- Similitud {h['score']:.2f}: {cond}"
            if code:
                line += f" | código='{code}'"
            line += f" | {desc}"
            if reason:
                line += f"\n    ⚠ Razón registrada: {reason}"
            return line

        context_lines = "\n".join(_format_context_line(h) for h in similar_images[:5])

        # Usar el system_instruction del usuario si existe; si no, el default
        sys_instr = system_instruction or self.DEFAULT_SYSTEM_INSTRUCTION
        temp      = temperature if temperature is not None else self.DEFAULT_TEMPERATURE

        rag_prompt = (
            f"Descripción de la imagen analizada:\n{description}\n\n"
            f"Código extraído por OCR: '{extracted_code}' "
            f"(confianza OCR: {confidence_ocr:.0%})\n\n"
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
            "extracted_code":       extracted_code,
            "confidence_ocr":       confidence_ocr,
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
