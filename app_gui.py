#!/usr/bin/env python3
"""
Interfaz gráfica (GUI) con Streamlit para el sistema RAG de cilindros.

Uso:
    streamlit run app_gui.py
    # o alternativamente:
    python -m streamlit run app_gui.py

Por defecto consume la API REST (api_server.py en localhost:5000).
Si la API no está disponible, cambia a modo "directo" en la barra lateral.
"""

from __future__ import annotations

import base64
import io
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# ── Configuración de página ──────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Cilindros",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API_URL debe apuntar al servicio rag-api:
#   - En Cloud Run GUI: URL pública del servicio rag-api
#     Ej: https://rag-api-xxxx-uc.a.run.app
#   - En local: http://localhost:5000
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:5000")
VALID_CONDITIONS = ["correct", "dented", "false_positive"]
CONDITION_LABELS = {
    "correct":        "✅ Correcto (buen estado)",
    "dented":         "🔴 Dañado (abolladuras)",
    "false_positive": "🟡 Falso positivo (parece dañado)",
}
CONDITION_COLORS = {
    "correct":        "#10b981",
    "dented":         "#ef4444",
    "false_positive": "#f59e0b",
}


# ── Capa de cliente (idéntica a rag_cli.py, encapsulada aquí) ───────────────

@st.cache_resource(show_spinner=False)
def get_client(api_url: str, mode: str):
    """Crea y cachea el cliente."""
    if mode == "Directo (sin API)":
        from image_processor import CylinderImageProcessor
        return _DirectClient(CylinderImageProcessor())
    return _ApiClient(api_url)


class _ApiClient:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.kind = "api"

    def _encode(self, file_bytes: bytes) -> str:
        return base64.b64encode(file_bytes).decode("utf-8")

    def _get(self, path: str) -> Dict[str, Any]:
        r = requests.get(f"{self.url}{path}", timeout=60)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.url}{path}", json=body, timeout=120)
        if r.status_code >= 400:
            try:
                err = r.json()
            except Exception:
                err = {"error": r.text}
            raise RuntimeError(err.get("error") or err)
        return r.json()

    def health(self)   -> Dict[str, Any]: return self._get("/health")
    def stats(self)    -> Dict[str, Any]: return self._get("/stats")["stats"]
    def upload(self, file_bytes: bytes, condition: str,
               confidence: float, verified: bool,
               system_instruction: Optional[str] = None,
               user_prompt: Optional[str] = None,
               reason: Optional[str] = None,
               temperature: Optional[float] = None) -> Dict[str, Any]:
        body = {
            "image_data":         self._encode(file_bytes),
            "cylinder_condition": condition,
            "confidence_score":   confidence,
            "source_info":        {"source": "gui", "verified": verified},
        }
        if system_instruction:
            body["system_instruction"] = system_instruction
        if user_prompt:
            body["user_prompt"] = user_prompt
        if reason:
            body["reason"] = reason
        if temperature is not None:
            body["temperature"] = temperature
        return self._post("/process-image", body)

    def classify(self, file_bytes: bytes, threshold: float,
                 system_instruction: Optional[str] = None,
                 user_prompt: Optional[str] = None,
                 temperature: Optional[float] = None) -> Dict[str, Any]:
        body = {
            "image_data":          self._encode(file_bytes),
            "confidence_threshold": threshold,
        }
        if system_instruction:
            body["system_instruction"] = system_instruction
        if user_prompt:
            body["user_prompt"] = user_prompt
        if temperature is not None:
            body["temperature"] = temperature
        return self._post("/classify-image", body)

    def search(self, file_bytes: bytes, limit: int,
               threshold: float, filter_cond: Optional[str],
               system_instruction: Optional[str] = None,
               user_prompt: Optional[str] = None,
               temperature: Optional[float] = None) -> Dict[str, Any]:
        body = {"image_data": self._encode(file_bytes), "limit": limit,
                "score_threshold": threshold}
        if filter_cond:
            body["filter_condition"] = filter_cond
        if system_instruction:
            body["system_instruction"] = system_instruction
        if user_prompt:
            body["user_prompt"] = user_prompt
        if temperature is not None:
            body["temperature"] = temperature
        return self._post("/search-similar", body)


class _DirectClient:
    def __init__(self, proc):
        self.proc = proc
        self.kind = "direct"

    def health(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "config": {
                "qdrant_type":     "cloud" if self.proc.qdrant_cloud_url else "local",
                "embedding_model": self.proc.embedding_model,
                "vision_model":    self.proc.vision_model_id,
                "vector_size":     self.proc.vector_size,
                "sdk":             "google-genai",
            }
        }

    def stats(self) -> Dict[str, Any]:
        info = self.proc.qdrant_client.get_collection(self.proc.collection_name)
        return {
            "total_images":    info.points_count,
            "vector_size":     info.config.params.vectors.size,
            "distance":        str(info.config.params.vectors.distance),
            "indexed_vectors": info.indexed_vectors_count,
            "qdrant_type":     "cloud" if self.proc.qdrant_cloud_url else "local",
            "embedding_model": self.proc.embedding_model,
            "vision_model":    self.proc.vision_model_id,
            "sdk":             "google-genai",
        }

    def upload(self, file_bytes: bytes, condition: str,
               confidence: float, verified: bool,
               system_instruction: Optional[str] = None,
               user_prompt: Optional[str] = None,
               reason: Optional[str] = None,
               temperature: Optional[float] = None) -> Dict[str, Any]:
        pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        pid = self.proc.upload_image(
            image=pil, cylinder_condition=condition,
            confidence_score=confidence, source="gui", verified=verified,
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            reason=reason,
            temperature=temperature,
        )
        return {"success": True, "point_id": pid}

    def classify(self, file_bytes: bytes, threshold: float,
                 system_instruction: Optional[str] = None,
                 user_prompt: Optional[str] = None,
                 temperature: Optional[float] = None) -> Dict[str, Any]:
        pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        result = self.proc.classify_cylinder(
            image=pil, confidence_threshold=threshold,
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        return {"success": True, "classification": result}

    def search(self, file_bytes: bytes, limit: int,
               threshold: float, filter_cond: Optional[str],
               system_instruction: Optional[str] = None,
               user_prompt: Optional[str] = None,
               temperature: Optional[float] = None) -> Dict[str, Any]:
        pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        hits = self.proc.search_similar_images(
            query_image=pil, limit=limit,
            score_threshold=threshold, filter_condition=filter_cond,
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        return {"success": True, "similar_images": hits, "count": len(hits)}


# ── Estilos CSS ──────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stat-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.2rem; border-radius: 0.75rem; color: #f1f5f9;
        border-left: 4px solid #3b82f6;
    }
    .stat-card .label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .stat-card .value { font-size: 1.8rem; font-weight: 700; margin-top: 0.3rem; color: #fff; }
    .condition-pill {
        display: inline-block; padding: 0.3rem 0.9rem; border-radius: 999px;
        font-weight: 600; font-size: 0.9rem; color: white;
    }
    .match-card {
        background: #0f172a; padding: 1rem; border-radius: 0.5rem;
        border: 1px solid #1e293b; margin-bottom: 0.5rem;
    }
    .match-card .score { font-size: 1.4rem; font-weight: 700; color: #60a5fa; }
    .stProgress > div > div > div > div { background-color: #3b82f6; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuración")

    api_mode = st.radio(
        "Modo de operación",
        ["API REST (recomendado)", "Directo (sin API)"],
        help="Si la API Flask no está corriendo, usa modo directo",
    )

    api_url = st.text_input(
        "URL de la API",
        value=DEFAULT_API_URL,
        disabled=api_mode == "Directo (sin API)",
    )

    st.divider()

    page = st.radio(
        "Navegación",
        ["📊 Dashboard", "📤 Subir imagen", "🔍 Clasificar",
         "🔎 Buscar similares", "📁 Subida por lote", "⚙️ Estado del sistema"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ── Cliente (cacheado) ───────────────────────────────────────────────────────

client = get_client(api_url, api_mode)


# ── Páginas ──────────────────────────────────────────────────────────────────

def page_dashboard():
    st.title("📊 Dashboard")
    st.caption("Resumen del sistema RAG de clasificación de cilindros")

    try:
        stats = client.stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total de imágenes",   stats.get("total_images", 0))
        c2.metric("Vectores indexados",  stats.get("indexed_vectors", 0))
        c3.metric("Dimensión",           stats.get("vector_size", "—"))
        c4.metric("Distancia",           stats.get("distance", "—"))

        st.divider()
        st.markdown("### 🔧 Configuración activa")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Modo de operación:** `{client.kind}`")
            st.markdown(f"**Tipo de Qdrant:** `{stats.get('qdrant_type', '—')}`")
            st.markdown(f"**SDK:** `{stats.get('sdk', '—')}`")
        with col2:
            st.markdown(f"**Modelo visión:** `{stats.get('vision_model', '—')}`")
            st.markdown(f"**Modelo embeddings:** `{stats.get('embedding_model', '—')}`")
            st.markdown(f"**API URL:** `{api_url if client.kind == 'api' else 'N/A (modo directo)'}`")

    except Exception as e:
        st.error(f"❌ No se pudo conectar: {e}")
        if client.kind == "api":
            st.info("💡 Tip: Asegúrate de que `api_server.py` esté corriendo, "
                    "o cambia a modo 'Directo' en la barra lateral.")


def page_upload():
    st.title("📤 Subir imagen de entrenamiento")
    st.caption("Sube una imagen etiquetada para entrenar el sistema RAG")

    # ── System instruction (FUERA del form) ─────────────────────────────────
    with st.expander("🧠 System instruction (opcional)", expanded=False):
        preset = st.selectbox(
            "Preset",
            list(PRESET_SYS_INSTRUCTIONS.keys()),
            key="upload_preset",
            help="Define cómo Gemini analiza la imagen al subirla",
        )
        if preset == "Personalizado":
            sys_instr_upload = st.text_area(
                "✏️ Escribe tu instrucción personalizada:",
                value=st.session_state.get("custom_sys_instr_upload", ""),
                height=150,
                key="custom_sys_instr_upload",
                placeholder="Ej: 'Detecta abolladuras mayores a 1cm...'",
            )
        else:
            preset_text = PRESET_SYS_INSTRUCTIONS[preset]
            sys_instr_upload = preset_text
            st.text_area(
                "Instrucción que se enviará a Gemini (solo lectura):",
                value=preset_text or "(vacío → se usará el default interno)",
                height=150,
                disabled=True,
            )
            if not preset_text:
                sys_instr_upload = None

        temperature_upload = st.slider(
            "Temperatura Gemini",
            min_value=0.0, max_value=1.0, value=0.2, step=0.05,
            key="upload_temperature",
        )

    col_form, col_preview = st.columns([1, 1])

    with col_form:
        with st.form("upload_form", clear_on_submit=True):
            condition = st.selectbox(
                "Condición del cilindro *",
                options=VALID_CONDITIONS,
                format_func=lambda x: CONDITION_LABELS[x],
            )

            confidence = st.slider(
                "Confianza de la etiqueta",
                min_value=0.0, max_value=1.0, value=1.0, step=0.05,
                help="1.0 = verificado por humano, menor = confianza parcial",
            )

            verified = st.checkbox(
                "Verificado por humano",
                value=True,
                help="Marca si un humano revisó esta etiqueta",
            )

            # ── Razón (CRÍTICO para falsos positivos / falsos negativos) ───
            reason = st.text_area(
                "📝 Razón / justificación de la etiqueta",
                value=st.session_state.get("upload_reason", ""),
                height=100,
                key="upload_reason",
                placeholder=(
                    "Ejemplos:\n"
                    "• 'Estampado confundido con abolladura → es falso positivo'\n"
                    "• 'Marca de fábrica, NO es daño'\n"
                    "• 'Abolladura real en zona superior izquierda, ~3cm'\n"
                    "• 'Cilindro en buen estado, superficie íntegra'"
                ),
                help=(
                    "Esta razón se guarda como metadata y se inyecta en el "
                    "contexto RAG al clasificar imágenes similares. Es "
                    "CLAVE para corregir falsos positivos y falsos negativos."
                ),
            )

            uploaded = st.file_uploader(
                "Imagen del cilindro *",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
            )

            submitted = st.form_submit_button("📤 Subir imagen", type="primary",
                                              use_container_width=True)

    with col_preview:
        st.markdown("### 👁️ Vista previa")
        if uploaded is None:
            st.info("👈 Selecciona una imagen para ver la vista previa aquí")
        else:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)
            st.caption(f"Tamaño: {uploaded.size / 1024:.1f} KB · "
                       f"Tipo: {uploaded.type}")

    if submitted:
        if uploaded is None:
            st.error("❌ Debes seleccionar una imagen")
            return
        try:
            with st.spinner("⏳ Procesando con Gemini y almacenando en Qdrant…"):
                t0 = time.time()
                data = client.upload(
                    uploaded.getvalue(), condition, confidence, verified,
                    system_instruction=sys_instr_upload if sys_instr_upload else None,
                    reason=reason.strip() if reason else None,
                    temperature=temperature_upload,
                )
                dt = time.time() - t0

            st.success(f"✅ Imagen subida correctamente en {dt:.1f}s")
            st.markdown(f"""
                <div class="match-card">
                    <div class="score">🆔 {data.get('point_id', '—')}</div>
                    <div style="margin-top:0.5rem;">
                        <span class="condition-pill" style="background:{CONDITION_COLORS[condition]};">
                            {CONDITION_LABELS[condition]}
                        </span>
                        <span style="margin-left:1rem; color:#94a3b8;">
                            Confianza: {confidence:.0%}
                        </span>
                    </div>
                    {f'<div style="margin-top:0.5rem; color:#94a3b8; font-size:0.9rem;">📝 {reason}</div>' if reason else ''}
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error: {e}")


# ── Presets de System Instruction para la GUI ────────────────────────────────

PRESET_SYS_INSTRUCTIONS = {
    "(usar default)": "",
    "Crosland (control calidad)": (
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
    ),
    "Estricto (abolladuras)": (
        "Eres un inspector MUY ESTRICTO de cilindros industriales. "
        "Considera CUALQUIER irregularidad en la superficie, "
        "por mínima que sea, como posible abolladura. "
        "Es preferible un falso positivo a pasar por alto un daño real. "
        "Busca específicamente: hundimientos, golpes, deformaciones, "
        "marcas de impacto, abolladuras pequeñas, medianas o grandes, "
        "y cualquier desviación respecto a una superficie cilíndrica perfecta."
    ),
    "Permisivo": (
        "Eres un inspector PERMISIVO. Solo marca como dañado si el daño "
        "es claramente visible y significativo. Marcas menores o sombras "
        "no deben clasificarse como abolladuras."
    ),
    "Solo daños evidentes": (
        "Eres un inspector moderado. Clasifica como 'dented' únicamente "
        "daños que serían rechazados en una inspección de calidad estándar. "
        "Daños cosméticos menores no cuentan."
    ),
    "Personalizado": "",
}


def page_classify():
    st.title("🔍 Clasificar imagen")
    st.caption("Sube una imagen y el sistema la clasificará usando RAG")

    # ── System instruction FUERA del form (necesario para edición interactiva) ──
    with st.expander("🧠 System instruction (opcional)", expanded=False):
        preset = st.selectbox(
            "Preset",
            list(PRESET_SYS_INSTRUCTIONS.keys()),
            key="classify_preset",
            help="Define cómo Gemini analiza la imagen",
        )

        # Si NO es personalizado, mostrar el texto del preset (solo lectura)
        # Si es personalizado, mostrar text_area editable
        if preset == "Personalizado":
            sys_instr = st.text_area(
                "✏️ Escribe tu instrucción personalizada:",
                value=st.session_state.get("custom_sys_instr_classify", ""),
                height=200,
                placeholder=(
                    "Ej: 'Detecta solo abolladuras mayores a 2cm y diferencia "
                    "marcas de fábrica de golpes. Ignora el estampado legítimo.'"
                ),
                key="custom_sys_instr_classify",
            )
        else:
            preset_text = PRESET_SYS_INSTRUCTIONS[preset]
            sys_instr = preset_text
            st.text_area(
                "Instrucción que se enviará a Gemini (solo lectura):",
                value=preset_text or "(vacío → se usará el default interno)",
                height=180,
                disabled=True,
            )
            if not preset_text:
                sys_instr = None  # usar default interno

        # Temperatura (default 0.2 = más determinístico)
        temperature = st.slider(
            "Temperatura Gemini",
            min_value=0.0, max_value=1.0, value=0.2, step=0.05,
            help="Bajo (0.0-0.3) = más determinístico. Alto (0.7-1.0) = más creativo.",
            key="classify_temperature",
        )

    # ── Form principal ──────────────────────────────────────────────────────
    col_form, col_preview = st.columns([1, 1])

    with col_form:
        with st.form("classify_form", clear_on_submit=False):
            threshold = st.slider(
                "Umbral de confianza",
                min_value=0.0, max_value=1.0, value=0.7, step=0.05,
                help="Si la confianza es menor, se marca como 'requiere revisión'",
                key="classify_threshold",
            )
            uploaded = st.file_uploader(
                "Imagen a clasificar",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="classify_uploader",
            )
            submitted = st.form_submit_button("🔍 Clasificar", type="primary",
                                              use_container_width=True)

    with col_preview:
        st.markdown("### 👁️ Imagen")
        if uploaded is None:
            st.info("👈 Selecciona una imagen")
        else:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)

    if submitted:
        if uploaded is None:
            st.error("❌ Debes seleccionar una imagen")
            return
        try:
            with st.spinner("⏳ Analizando imagen y buscando similares…"):
                t0 = time.time()
                data = client.classify(
                    uploaded.getvalue(),
                    threshold,
                    system_instruction=sys_instr if sys_instr else None,
                    temperature=temperature,
                )
                dt = time.time() - t0
            render_classification(data, dt)

            # ── Debug crudo (URL + body + response) ──────────────────────
            with st.expander("🔧 Debug — request/response crudos", expanded=False):
                # Detectar la URL según el modo de operación
                if isinstance(client, _ApiClient):
                    endpoint = f"{client.url}/classify-image"
                    st.markdown(f"**Endpoint:** `{endpoint}`")
                    st.markdown("**Body enviado:**")
                    st.json({
                        "image_data":           "<base64 omitido>",
                        "confidence_threshold":  threshold,
                        "system_instruction":    sys_instr if sys_instr else None,
                        "temperature":           temperature,
                    })
                else:
                    st.markdown("**Endpoint:** _(modo directo — sin HTTP)_")
                    st.markdown("**Parámetros internos:**")
                    st.json({
                        "confidence_threshold":  threshold,
                        "system_instruction":    sys_instr if sys_instr else None,
                        "temperature":           temperature,
                    })
                st.markdown("**Response recibida:**")
                st.json(data)
        except Exception as e:
            st.error(f"❌ Error: {e}")


def render_classification(data: Dict[str, Any], dt: float):
    if not data.get("success"):
        st.error("❌ La clasificación falló")
        return

    cls = data["classification"]
    cond = cls["predicted_condition"]
    conf = cls["confidence"]
    is_conf = cls["is_confident"]

    st.markdown(f"""
        <div class="match-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="condition-pill" style="background:{CONDITION_COLORS.get(cond, '#6b7280')}; font-size:1.1rem; padding:0.5rem 1.2rem;">
                    {CONDITION_LABELS.get(cond, cond)}
                </span>
                <span class="score">{conf:.1%}</span>
            </div>
            <div style="margin-top:0.8rem; color:#94a3b8; font-size:0.9rem;">
                {'✅ Clasificación confiable' if is_conf else '⚠️ Requiere revisión humana'} ·
                {cls['similar_images_count']} similares ·
                mejor match {cls['best_match_score']:.3f} · {dt:.1f}s
            </div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 📝 Descripción (Gemini Vision)")
        st.info(cls.get("description", "—"))

        st.markdown("### 🧠 Análisis RAG (Gemini + contexto Qdrant)")
        st.success(cls.get("rag_explanation", "—"))

    with c2:
        st.markdown("### 🗳️ Votación ponderada")
        scores = cls.get("condition_scores", {})
        if scores:
            total = sum(scores.values()) or 1
            for k, v in sorted(scores.items(), key=lambda x: -x[1]):
                pct = v / total
                color = CONDITION_COLORS.get(k, "#6b7280")
                st.markdown(f"**{CONDITION_LABELS.get(k, k)}**")
                st.progress(min(pct, 1.0))
                st.caption(f"{pct:.1%} ({v:.3f})")


def page_search():
    st.title("🔎 Buscar imágenes similares")
    st.caption("Encuentra imágenes visualmente parecidas en la base de datos")

    with st.form("search_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            limit = st.number_input("Máx. resultados", 1, 50, 5)
        with col2:
            threshold = st.slider("Similitud mínima", 0.0, 1.0, 0.5, 0.05)
        with col3:
            filter_cond = st.selectbox(
                "Filtrar por condición",
                options=["(todas)"] + VALID_CONDITIONS,
                format_func=lambda x: CONDITION_LABELS.get(x, x),
            )

        # System instruction opcional también en búsqueda
        sys_instr_search = st.text_input(
            "🧠 System instruction (opcional)",
            value=st.session_state.get("custom_sys_instr_search", ""),
            placeholder="Deja vacío para usar el default",
            help="Personaliza cómo Gemini analiza la imagen antes de buscar similares",
        )
        st.session_state["custom_sys_instr_search"] = sys_instr_search

        uploaded = st.file_uploader(
            "Imagen de consulta",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
        submitted = st.form_submit_button("🔎 Buscar", type="primary")

    if uploaded is not None:
        st.image(uploaded, caption="Imagen de consulta", width=300)

    if submitted:
        if uploaded is None:
            st.error("❌ Debes seleccionar una imagen")
            return
        try:
            with st.spinner("⏳ Buscando similares…"):
                fcond = None if filter_cond == "(todas)" else filter_cond
                data = client.search(
                    uploaded.getvalue(), limit, threshold, fcond,
                    system_instruction=sys_instr_search if sys_instr_search else None,
                )
            render_search_results(data)
        except Exception as e:
            st.error(f"❌ Error: {e}")


def render_search_results(data: Dict[str, Any]):
    hits = data.get("similar_images", [])
    if not hits:
        st.warning("⚠️ No se encontraron imágenes similares con esos criterios")
        return

    st.success(f"✅ {len(hits)} imágenes similares encontradas")

    for i, h in enumerate(hits, 1):
        meta = h["metadata"]
        cond = meta["cylinder_condition"]
        with st.expander(
            f"#{i} · Score {h['score']:.3f} · "
            f"{CONDITION_LABELS.get(cond, cond)}",
            expanded=(i <= 3),
        ):
            c1, c2 = st.columns([3, 1])
            with c1:
                if meta.get("description"):
                    st.markdown("**Descripción:**")
                    st.write(meta["description"])
                st.markdown(f"**ID:** `{h['id']}`")
                st.markdown(f"**Fecha:** {meta.get('upload_timestamp', '—')}")
                st.markdown(f"**Fuente:** `{meta.get('source', '—')}`")
                st.markdown(f"**Verificado:** {'✅' if meta.get('verified') else '❌'}")
            with c2:
                st.markdown(f"""
                    <div class="match-card" style="text-align:center;">
                        <div class="score">{h['score']:.3f}</div>
                        <div style="font-size:0.8rem; color:#94a3b8;">similitud</div>
                    </div>
                """, unsafe_allow_html=True)


def page_batch():
    st.title("📁 Subida por lote")
    st.caption("Sube múltiples imágenes con la misma etiqueta")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Parámetros")
        condition = st.selectbox(
            "Condición para todas las imágenes",
            options=VALID_CONDITIONS,
            format_func=lambda x: CONDITION_LABELS[x],
            key="batch_cond",
        )
        confidence = st.slider("Confianza", 0.0, 1.0, 1.0, 0.05, key="batch_conf")
        verified   = st.checkbox("Verificado por humano", value=True, key="batch_verified")

    with col2:
        st.markdown("### Imágenes")
        files = st.file_uploader(
            "Selecciona una o varias imágenes",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
        )
        if files:
            st.info(f"📎 {len(files)} imágenes seleccionadas")

    if st.button("📤 Subir todas", type="primary", disabled=not files):
        if not files:
            st.error("❌ Selecciona al menos una imagen")
            return

        progress = st.progress(0.0, text="Iniciando…")
        ok = fail = 0
        errors: List[str] = []
        results_container = st.container()

        for i, f in enumerate(files):
            try:
                data = client.upload(f.getvalue(), condition, confidence, verified)
                if data.get("success"):
                    ok += 1
                else:
                    fail += 1
                    errors.append(f"{f.name}: {data.get('error', '?')}")
            except Exception as e:
                fail += 1
                errors.append(f"{f.name}: {e}")

            pct = (i + 1) / len(files)
            progress.progress(pct, text=f"Procesando {i+1}/{len(files)}…")

        progress.empty()
        c1, c2 = st.columns(2)
        c1.success(f"✅ Exitosas: {ok}")
        if fail:
            c2.error(f"❌ Fallidas: {fail}")
            with results_container:
                st.markdown("**Errores:**")
                for err in errors:
                    st.write(f"- {err}")
        else:
            c2.info("🎉 Todas las imágenes se subieron correctamente")


def page_status():
    st.title("⚙️ Estado del sistema")
    st.caption("Información técnica y diagnóstico")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🏥 Health Check")
        if st.button("🔄 Probar conexión", type="primary"):
            st.session_state["_health"] = None
        try:
            with st.spinner("Probando…"):
                h = client.health()
            st.session_state["_health"] = h
            st.success(f"✅ Estado: `{h.get('status', '—')}`")
            cfg = h.get("config", {})
            for k, v in cfg.items():
                st.markdown(f"- **{k}:** `{v}`")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    with c2:
        st.markdown("### 📊 Estadísticas detalladas")
        try:
            stats = client.stats()
            for k, v in stats.items():
                st.markdown(f"- **{k}:** `{v}`")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    st.divider()
    st.markdown("### ℹ️ Información de uso")
    st.markdown(f"""
        - **Modo actual:** `{client.kind}`
        - **API URL:** `{api_url}`
        - **SDK Gemini:** `google-genai`
        - **Modelos:** `gemini-2.5-flash` (visión) · `gemini-embedding-2` (embeddings)
        - **Vector size:** 3072

        **Atajos:**
        - `Ctrl+R` / `Cmd+R` → recargar la página
        - Cambia modo/API URL en la barra lateral izquierda
    """)


# ── Router ───────────────────────────────────────────────────────────────────

PAGES = {
    "📊 Dashboard":            page_dashboard,
    "📤 Subir imagen":         page_upload,
    "🔍 Clasificar":           page_classify,
    "🔎 Buscar similares":     page_search,
    "📁 Subida por lote":      page_batch,
    "⚙️ Estado del sistema":   page_status,
}

PAGES[page]()
