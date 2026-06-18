#!/usr/bin/env python3
"""
Interfaz CLI / menú interactivo para el sistema RAG de cilindros.

Dos modos de uso:

  • Modo comando (automatización / scripts):
      python rag_cli.py upload   imagen.jpg correct
      python rag_cli.py classify imagen.jpg
      python rag_cli.py search   imagen.jpg --limit 5
      python rag_cli.py stats
      python rag_cli.py health
      python rag_cli.py batch    ./imgs/ dented --recursive

  • Modo interactivo (menú guiado):
      python rag_cli.py menu
      python rag_cli.py          (sin argumentos → abre el menú)

Por defecto usa la API REST (api_server.py en localhost:5000).
Para usar el procesador directo (sin API), pasar --direct.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Constantes / configuración ──────────────────────────────────────────────

VALID_CONDITIONS = ("correct", "dented", "false_positive")
DEFAULT_API_URL  = os.getenv("API_URL", "http://localhost:5000")
SUPPORTED_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Colores ANSI (se desactivan automáticamente si no es TTY o NO_COLOR está set)
USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def c(code: str, text: str) -> str:
    """Aplicar color ANSI si está habilitado."""
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(t: str)   -> str: return c("1",  t)
def green(t: str)  -> str: return c("32", t)
def red(t: str)    -> str: return c("31", t)
def yellow(t: str) -> str: return c("33", t)
def blue(t: str)   -> str: return c("34", t)
def cyan(t: str)   -> str: return c("36", t)
def gray(t: str)   -> str: return c("90", t)


# ── Capa de transporte: API REST vs procesador directo ──────────────────────

class RagClient:
    """
    Wrapper que abstrae el origen de los datos:
      - HTTP → llama a api_server.py
      - DIRECT → usa image_processor.py directamente
    """

    def __init__(self, api_url: str = DEFAULT_API_URL, direct: bool = False):
        self.api_url = api_url.rstrip("/")
        self.direct  = direct
        self._proc   = None  # cache del procesador directo

    def _processor(self):
        if self._proc is None:
            from image_processor import CylinderImageProcessor  # import perezoso
            self._proc = CylinderImageProcessor()
        return self._proc

    # ── Operaciones ───────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        if self.direct:
            p = self._processor()
            return {
                "status": "healthy",
                "config": {
                    "qdrant_type":     "cloud" if p.qdrant_cloud_url else "local",
                    "embedding_model": p.embedding_model,
                    "vision_model":    p.vision_model_id,
                    "vector_size":     p.vector_size,
                    "sdk":             "google-genai",
                }
            }
        return self._get("/health")

    def stats(self) -> Dict[str, Any]:
        if self.direct:
            p     = self._processor()
            info  = p.qdrant_client.get_collection(p.collection_name)
            return {
                "total_images":     info.points_count,
                "vector_size":      info.config.params.vectors.size,
                "distance":         str(info.config.params.vectors.distance),
                "indexed_vectors":  info.indexed_vectors_count,
                "qdrant_type":      "cloud" if p.qdrant_cloud_url else "local",
                "embedding_model":  p.embedding_model,
                "vision_model":     p.vision_model_id,
                "sdk":              "google-genai",
            }
        return self._get("/stats")["stats"]

    def upload(self, image_path: str, condition: str,
               confidence: float = 1.0, verified: bool = True) -> Dict[str, Any]:
        if condition not in VALID_CONDITIONS:
            raise ValueError(f"Condición inválida. Usa una de: {VALID_CONDITIONS}")

        if self.direct:
            point_id = self._processor().upload_image(
                image=image_path,
                cylinder_condition=condition,
                confidence_score=confidence,
                source="cli",
                verified=verified,
            )
            return {"success": True, "point_id": point_id}

        return self._post("/process-image", {
            "image_data":        self._encode(image_path),
            "cylinder_condition": condition,
            "confidence_score":  confidence,
            "source_info":       {"source": "cli", "verified": verified},
        })

    def classify(self, image_path: str, threshold: float = 0.7) -> Dict[str, Any]:
        if self.direct:
            result = self._processor().classify_cylinder(
                image=image_path,
                confidence_threshold=threshold,
            )
            return {"success": True, "classification": result}

        return self._post("/classify-image", {
            "image_data":          self._encode(image_path),
            "confidence_threshold": threshold,
        })

    def search(self, image_path: str, limit: int = 5,
               threshold: float = 0.5,
               filter_condition: Optional[str] = None) -> Dict[str, Any]:
        if self.direct:
            results = self._processor().search_similar_images(
                query_image=image_path,
                limit=limit,
                score_threshold=threshold,
                filter_condition=filter_condition,
            )
            return {"success": True, "similar_images": results, "count": len(results)}

        body = {"image_data": self._encode(image_path), "limit": limit,
                "score_threshold": threshold}
        if filter_condition:
            body["filter_condition"] = filter_condition
        return self._post("/search-similar", body)

    # ── Helpers HTTP / imagen ─────────────────────────────────────────────

    def _encode(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get(self, path: str) -> Dict[str, Any]:
        r = requests.get(f"{self.api_url}{path}", timeout=60)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.api_url}{path}", json=body, timeout=120)
        if r.status_code >= 400:
            try:
                err = r.json()
            except Exception:
                err = {"error": r.text}
            raise RuntimeError(f"[{r.status_code}] {err.get('error') or err}")
        return r.json()


# ── Formateo de salida ──────────────────────────────────────────────────────

def print_health(data: Dict[str, Any]) -> None:
    print(bold(green("\n✓ API saludable")))
    print(json.dumps(data, indent=2, ensure_ascii=False))


def print_stats(data: Dict[str, Any]) -> None:
    print(bold(cyan("\n📊 Estadísticas de la base de datos\n")))
    rows = [
        ("Total de imágenes",     data.get("total_images", 0)),
        ("Vectores indexados",    data.get("indexed_vectors", 0)),
        ("Tamaño de vector",      data.get("vector_size", "—")),
        ("Distancia",             data.get("distance", "—")),
        ("Tipo de Qdrant",        data.get("qdrant_type", "—")),
        ("Modelo embeddings",     data.get("embedding_model", "—")),
        ("Modelo visión",         data.get("vision_model", "—")),
        ("SDK",                   data.get("sdk", "—")),
    ]
    label_w = max(len(k) for k, _ in rows) + 2
    for k, v in rows:
        print(f"  {gray(k.ljust(label_w))}{bold(str(v))}")


def print_upload(data: Dict[str, Any], image_path: str, condition: str) -> None:
    if data.get("success"):
        print(bold(green(f"\n✓ Imagen subida")))
        print(f"  {gray('archivo:    ')}{image_path}")
        print(f"  {gray('condición:  ')}{bold(condition)}")
        print(f"  {gray('point_id:   ')}{cyan(data.get('point_id', '—'))}")
    else:
        print(bold(red(f"\n✗ Error: {data.get('error', 'desconocido')}")))
        sys.exit(1)


def print_classification(data: Dict[str, Any], image_path: str) -> None:
    if not data.get("success"):
        print(bold(red(f"\n✗ Error: {data.get('error', 'desconocido')}")))
        sys.exit(1)

    c_ = data["classification"]
    cond      = c_["predicted_condition"]
    conf      = c_["confidence"]
    confident = c_["is_confident"]

    icon = "🟢" if confident else "🟡"
    print(bold(f"\n{icon} Clasificación: ") +
          bold(green(cond) if cond == "correct" else
               yellow(cond) if cond == "dented" else
               cyan(cond)   if cond == "false_positive" else
               red(cond)))
    print(f"  {gray('archivo:        ')}{image_path}")
    print(f"  {gray('confianza:      ')}{bold(f'{conf:.2%}')}")
    print(f"  {gray('confiable:      ')}{'sí' if confident else 'no (revisión humana)'}")
    print(f"  {gray('similares:      ')}{c_['similar_images_count']}")
    print(f"  {gray('mejor match:    ')}{c_['best_match_score']:.3f}")

    if c_.get("description"):
        desc = c_["description"]
        print(f"\n  {bold('Descripción:')}\n  {gray(desc)}")

    if c_.get("rag_explanation"):
        print(f"\n  {bold('Análisis RAG:')}\n  {cyan(c_['rag_explanation'])}")

    if c_.get("condition_scores"):
        scores = c_["condition_scores"]
        print(f"\n  {bold('Votación ponderada:')}")
        for k, v in sorted(scores.items(), key=lambda x: -x[1]):
            bar = "█" * int(v * 30)
            print(f"    {k.ljust(15)} {yellow(bar)} {v:.3f}")


def print_search(data: Dict[str, Any], image_path: str) -> None:
    if not data.get("success"):
        print(bold(red(f"\n✗ Error: {data.get('error', 'desconocido')}")))
        sys.exit(1)

    hits: List[Dict[str, Any]] = data["similar_images"]
    print(bold(cyan(f"\n🔍 {len(hits)} imágenes similares a: {image_path}\n")))

    for i, h in enumerate(hits, 1):
        meta = h["metadata"]
        print(f"  {bold(f'#{i}')}  {gray('score:')} {h['score']:.3f}  "
              f"{gray('condición:')} {bold(meta['cylinder_condition'])}")
        if meta.get("description"):
            print(f"      {gray(meta['description'][:120])}")
        print()


# ── Subcomandos CLI ─────────────────────────────────────────────────────────

def cmd_upload(args, client: RagClient):
    print(gray(f"⏳ Procesando {args.image} → {args.condition}…"))
    data = client.upload(args.image, args.condition,
                         confidence=args.confidence,
                         verified=args.verified)
    print_upload(data, args.image, args.condition)


def cmd_classify(args, client: RagClient):
    print(gray(f"⏳ Clasificando {args.image}…"))
    data = client.classify(args.image, threshold=args.threshold)
    print_classification(data, args.image)


def cmd_search(args, client: RagClient):
    print(gray(f"⏳ Buscando similares a {args.image}…"))
    data = client.search(args.image, limit=args.limit,
                         threshold=args.threshold,
                         filter_condition=args.filter)
    print_search(data, args.image)


def cmd_stats(args, client: RagClient):
    print_stats(client.stats())


def cmd_health(args, client: RagClient):
    print_health(client.health())


def cmd_batch(args, client: RagClient):
    folder = Path(args.folder)
    if not folder.is_dir():
        print(bold(red(f"✗ Carpeta no encontrada: {folder}")))
        sys.exit(1)

    if args.recursive:
        files = [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    else:
        files = [p for p in folder.iterdir()
                 if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]

    if not files:
        print(yellow(f"⚠ No se encontraron imágenes en {folder}"))
        return

    print(bold(f"\n📁 Encontradas {len(files)} imágenes → condición: {args.condition}\n"))

    ok = fail = 0
    for i, f in enumerate(files, 1):
        try:
            data = client.upload(str(f), args.condition,
                                 confidence=args.confidence,
                                 verified=args.verified)
            if data.get("success"):
                ok += 1
                print(f"  {green('✓')} [{i}/{len(files)}] {f.name}")
            else:
                fail += 1
                print(f"  {red('✗')} [{i}/{len(files)}] {f.name}: "
                      f"{data.get('error', '?')}")
        except Exception as e:
            fail += 1
            print(f"  {red('✗')} [{i}/{len(files)}] {f.name}: {e}")

    print(bold(f"\n📊 Resultado: {green(ok)} ok, {red(fail)} fallos"))


# ── Modo interactivo ────────────────────────────────────────────────────────

def interactive_menu(client: RagClient):
    print(bold(cyan("\n╔════════════════════════════════════════════════════╗")))
    print(bold(cyan("║   RAG Cilindros — Menú Interactivo                ║")))
    print(bold(cyan("╚════════════════════════════════════════════════════╝")))
    print(gray(f"  Modo: {'API REST' if not client.direct else 'Directo (processor)'}\n"))

    actions = {
        "1": ("📤  Subir imagen de entrenamiento",     action_upload),
        "2": ("🔍  Clasificar imagen (RAG)",           action_classify),
        "3": ("🔎  Buscar imágenes similares",         action_search),
        "4": ("📊  Ver estadísticas de la base",       action_stats),
        "5": ("🏥  Health check del sistema",          action_health),
        "6": ("📁  Subir carpeta completa (batch)",    action_batch),
        "0": ("🚪  Salir",                             None),
    }

    while True:
        print()
        for k, (label, _) in actions.items():
            print(f"  {bold(cyan(k))}  {label}")
        print()
        choice = input(bold("Elige una opción: ")).strip()

        if choice == "0":
            print(gray("\n👋 Hasta luego\n"))
            break

        action = actions.get(choice)
        if action is None or action[1] is None:
            print(yellow("⚠ Opción no válida"))
            continue

        try:
            action[1](client)
        except KeyboardInterrupt:
            print(yellow("\n⚠ Operación cancelada"))
        except Exception as e:
            print(bold(red(f"\n✗ Error: {e}")))


def ask_image() -> str:
    while True:
        path = input("  Ruta de la imagen: ").strip().strip('"').strip("'")
        if not path:
            print(yellow("⚠ Ruta vacía"))
            continue
        if not Path(path).exists():
            print(yellow(f"⚠ No existe: {path}"))
            continue
        return path


def ask_condition() -> str:
    print("  Condición:")
    print(f"    {bold('1')} correct        (buen estado)")
    print(f"    {bold('2')} dented         (con abolladuras)")
    print(f"    {bold('3')} false_positive (parece dañado, está bien)")
    while True:
        c_ = input("  Elige: ").strip()
        if c_ in ("1", "correct"):        return "correct"
        if c_ in ("2", "dented"):         return "dented"
        if c_ in ("3", "false_positive"): return "false_positive"
        print(yellow("⚠ Opción inválida"))


def action_upload(client: RagClient):
    print(bold("\n📤 Subir imagen de entrenamiento\n"))
    img = ask_image()
    cond = ask_condition()
    conf = input("  Confianza [1.0]: ").strip()
    conf = float(conf) if conf else 1.0
    print(gray("⏳ Subiendo…"))
    data = client.upload(img, cond, confidence=conf, verified=True)
    print_upload(data, img, cond)


def action_classify(client: RagClient):
    print(bold("\n🔍 Clasificar imagen\n"))
    img = ask_image()
    print(gray("⏳ Clasificando…"))
    data = client.classify(img, threshold=0.7)
    print_classification(data, img)


def action_search(client: RagClient):
    print(bold("\n🔎 Buscar similares\n"))
    img = ask_image()
    limit = input("  ¿Cuántos resultados? [5]: ").strip()
    limit = int(limit) if limit else 5
    print(gray("⏳ Buscando…"))
    data = client.search(img, limit=limit, threshold=0.5)
    print_search(data, img)


def action_stats(client: RagClient):
    print_stats(client.stats())


def action_health(client: RagClient):
    print_health(client.health())


def action_batch(client: RagClient):
    print(bold("\n📁 Subir carpeta completa\n"))
    folder = input("  Ruta de la carpeta: ").strip().strip('"').strip("'")
    if not Path(folder).is_dir():
        print(red(f"✗ No existe: {folder}"))
        return
    cond = ask_condition()
    rec  = input("  ¿Recursivo? (s/N): ").strip().lower() == "s"
    print(gray("⏳ Procesando carpeta…"))

    args = argparse.Namespace(folder=folder, condition=cond,
                              confidence=1.0, verified=True, recursive=rec)
    cmd_batch(args, client)


# ── Parser principal ────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag_cli",
        description="Interfaz CLI para el sistema RAG de clasificación de cilindros",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python rag_cli.py health
  python rag_cli.py upload imagen.jpg correct
  python rag_cli.py classify imagen.jpg
  python rag_cli.py search imagen.jpg --limit 10
  python rag_cli.py batch ./dataset/ dented --recursive
  python rag_cli.py --direct classify imagen.jpg
  python rag_cli.py menu
        """,
    )
    p.add_argument("--api-url", default=DEFAULT_API_URL,
                   help=f"URL base de la API (default: {DEFAULT_API_URL})")
    p.add_argument("--direct", action="store_true",
                   help="Usar el procesador directo sin pasar por la API REST")
    p.add_argument("--no-color", action="store_true",
                   help="Desactivar colores en la salida")

    sub = p.add_subparsers(dest="cmd")

    # upload
    s = sub.add_parser("upload", help="Subir imagen con etiqueta de entrenamiento")
    s.add_argument("image", help="Ruta de la imagen")
    s.add_argument("condition", choices=VALID_CONDITIONS,
                   help="Condición del cilindro")
    s.add_argument("--confidence", type=float, default=1.0,
                   help="Confianza de la etiqueta (0.0-1.0, default: 1.0)")
    s.add_argument("--verified", action="store_true", default=True,
                   help="Marcar como verificado por humano")
    s.set_defaults(func=cmd_upload)

    # classify
    s = sub.add_parser("classify", help="Clasificar imagen usando RAG")
    s.add_argument("image", help="Ruta de la imagen a clasificar")
    s.add_argument("--threshold", type=float, default=0.7,
                   help="Umbral de confianza (0.0-1.0, default: 0.7)")
    s.set_defaults(func=cmd_classify)

    # search
    s = sub.add_parser("search", help="Buscar imágenes similares")
    s.add_argument("image", help="Ruta de la imagen de consulta")
    s.add_argument("--limit", type=int, default=5, help="Máx. resultados")
    s.add_argument("--threshold", type=float, default=0.5, help="Similitud mínima")
    s.add_argument("--filter", choices=VALID_CONDITIONS,
                   help="Filtrar por condición específica")
    s.set_defaults(func=cmd_search)

    # batch
    s = sub.add_parser("batch", help="Subir carpeta completa de imágenes")
    s.add_argument("folder", help="Ruta de la carpeta")
    s.add_argument("condition", choices=VALID_CONDITIONS, help="Etiqueta común")
    s.add_argument("--confidence", type=float, default=1.0)
    s.add_argument("--verified", action="store_true", default=True)
    s.add_argument("--recursive", action="store_true",
                   help="Buscar imágenes en subcarpetas")
    s.set_defaults(func=cmd_batch)

    # stats
    s = sub.add_parser("stats", help="Ver estadísticas de la base de datos")
    s.set_defaults(func=cmd_stats)

    # health
    s = sub.add_parser("health", help="Health check del sistema")
    s.set_defaults(func=cmd_health)

    # menu
    sub.add_parser("menu", help="Abrir menú interactivo")

    return p


def main():
    global USE_COLOR
    args = build_parser().parse_args()

    if args.no_color:
        USE_COLOR = False

    client = RagClient(api_url=args.api_url, direct=args.direct)

    # Sin argumentos → menú
    if args.cmd is None:
        interactive_menu(client)
        return

    if args.cmd == "menu":
        interactive_menu(client)
        return

    args.func(args, client)


if __name__ == "__main__":
    main()
