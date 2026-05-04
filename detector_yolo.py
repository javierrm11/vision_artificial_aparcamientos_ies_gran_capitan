"""
detector.py
────────────────────────────────────────────────
Detector de plazas de aparcamiento con YOLOv8.
Acepta imagen estática, webcam o archivo de vídeo.

Uso:
    # Imagen estática
    python detector.py --fuente imgs/foto.png

    # Webcam (índice 0 por defecto)
    python detector.py

    # Webcam específica
    python detector.py --fuente 1

    # Archivo de vídeo (.webm, .mp4, ...)
    python detector.py --fuente imgs/video.webm

Opciones:
    --conf   0.35     Confianza mínima YOLO (0-1)
    --cada   5        Analizar 1 de cada N frames (sólo vídeo/webcam)
    --visual          Guardar imagen anotada (sólo modo imagen)
    --spots  ruta     Ruta al spots.json (def: imgs/spots.json)

Controles (modo vídeo/webcam):
    G        → guardar captura actual (imagen + JSON)
    Q / ESC  → salir
────────────────────────────────────────────────
"""

import cv2
import json
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# ── Configuración por defecto ─────────────────
SPOTS_JSON  = Path("imgs/spots.json")
OUTPUT_DIR  = Path("imgs/capturas")
CONF_UMBRAL = 0.35
CADA_N      = 5
CLASES_VEH  = {2, 3, 5, 7}   # car, motorcycle, bus, truck (COCO)
# ─────────────────────────────────────────────

COL_LIBRE   = (0, 220, 80)
COL_OCUPADA = (0, 60, 220)
COL_YOLO    = (0, 165, 255)


# ══════════════════════════════════════════════
# Utilidades comunes
# ══════════════════════════════════════════════

def cargar_spots(path: Path):
    with open(path) as f:
        return json.load(f)


def punto_en_poligono(cx, cy, puntos):
    pts = np.array(puntos, dtype=np.float32)
    return cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0


def detectar(model, frame, conf):
    """Ejecuta YOLO sobre un frame y devuelve centros y bboxes de vehículos."""
    results = model(frame, conf=conf, classes=list(CLASES_VEH), verbose=False)[0]
    centros, boxes = [], []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        centros.append((cx, cy))
        boxes.append((x1, y1, x2, y2))
    return centros, boxes


def calcular_resultados(spots, centros):
    """Decide LIBRE/OCUPADA por plaza según los centros detectados."""
    resultados = []
    for s in spots:
        coches = [(cx, cy) for cx, cy in centros
                  if punto_en_poligono(cx, cy, s["points"])]
        resultados.append({
            "id":            s["id"],
            "estado":        "OCUPADA" if coches else "LIBRE",
            "coches_dentro": len(coches)
        })
    return resultados


def dibujar(frame, spots, resultados, boxes):
    """Dibuja plazas, bboxes YOLO y HUD sobre el frame."""
    overlay = frame.copy()

    for s, r in zip(spots, resultados):
        color  = COL_LIBRE if r["estado"] == "LIBRE" else COL_OCUPADA
        pts_np = np.array(s["points"], dtype=np.int32)
        cv2.fillPoly(overlay, [pts_np], color)
        cv2.polylines(frame, [pts_np], isClosed=True, color=color, thickness=2)
        cx = int(pts_np[:, 0].mean())
        cy = int(pts_np[:, 1].mean())
        cv2.putText(frame, f"{s['id']}:{'L' if r['estado']=='LIBRE' else 'O'}",
                    (cx - 14, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_YOLO, 1)
        cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 5, (0, 0, 255), -1)

    # HUD superior
    libres   = sum(1 for r in resultados if r["estado"] == "LIBRE")
    ocupadas = len(resultados) - libres
    cv2.rectangle(frame, (0, 0), (360, 34), (30, 30, 30), -1)
    cv2.putText(frame,
                f"Libres: {libres}/{len(resultados)}  Ocupadas: {ocupadas}/{len(resultados)}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return frame


def dibujar_leyenda_video(frame):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "[G] guardar captura   [Q/ESC] salir",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def guardar_captura(resultados, fuente, frame=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    libres   = [r for r in resultados if r["estado"] == "LIBRE"]
    ocupadas = [r for r in resultados if r["estado"] == "OCUPADA"]
    datos = {
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "fuente":       str(fuente),
        "total_plazas": len(resultados),
        "libres":       len(libres),
        "ocupadas":     len(ocupadas),
        "plazas":       resultados
    }

    json_path = OUTPUT_DIR / f"estado_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)

    if frame is not None:
        img_path = OUTPUT_DIR / f"captura_{ts}.png"
        cv2.imwrite(str(img_path), frame)
        print(f"✓ Guardado: {img_path.name}  +  {json_path.name}")
    else:
        print(f"✓ JSON guardado: {json_path.name}")

    print(f"  🟢 Libres: {len(libres)}  🔴 Ocupadas: {len(ocupadas)}")
    for r in resultados:
        icono = "🟢" if r["estado"] == "LIBRE" else "🔴"
        print(f"  {icono} Plaza {r['id']:>2}  {r['estado']}")


# ══════════════════════════════════════════════
# Modos de ejecución
# ══════════════════════════════════════════════

def modo_imagen(model, path: Path, spots, conf, guardar_visual):
    print(f"▸ Modo IMAGEN: {path}")
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"No se encontró: {path}")

    centros, boxes = detectar(model, frame, conf)
    print(f"  → {len(centros)} vehículos detectados")

    resultados = calcular_resultados(spots, centros)
    frame_viz  = dibujar(frame.copy(), spots, resultados, boxes)

    guardar_captura(resultados, path, frame=frame_viz if guardar_visual else None)

    # Mostrar siempre la ventana en modo imagen
    cv2.imshow("Detector aparcamiento — imagen", frame_viz)
    print("  Pulsa cualquier tecla para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def modo_video(model, fuente, spots, conf, cada):
    src  = int(fuente) if str(fuente).isdigit() else str(fuente)
    tipo = "WEBCAM" if isinstance(src, int) else f"VÍDEO ({fuente})"
    print(f"▸ Modo {tipo}")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {fuente}")

    resultados  = [{"id": s["id"], "estado": "...", "coches_dentro": 0} for s in spots]
    boxes       = []
    frame_count = 0

    print("▸ Detección activa. [G]=guardar  [Q/ESC]=salir\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1

        if frame_count % cada == 0:
            centros, boxes = detectar(model, frame, conf)
            resultados     = calcular_resultados(spots, centros)

        frame_viz = dibujar(frame.copy(), spots, resultados, boxes)
        frame_viz = dibujar_leyenda_video(frame_viz)
        cv2.imshow("Detector aparcamiento", frame_viz)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('g'), ord('G')):
            guardar_captura(resultados, fuente, frame=frame_viz)

    cap.release()
    cv2.destroyAllWindows()
    print("✓ Detector cerrado")


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

EXTENSIONES_IMAGEN = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
EXTENSIONES_VIDEO  = {".webm", ".mp4", ".avi", ".mov", ".mkv"}


def detectar_modo(fuente: str) -> str:
    sufijo = Path(fuente).suffix.lower()
    if sufijo in EXTENSIONES_IMAGEN:
        return "imagen"
    if sufijo in EXTENSIONES_VIDEO:
        return "video"
    return "video"   # webcam (número) o stream


def main():
    parser = argparse.ArgumentParser(
        description="Detector de aparcamiento — imagen / webcam / vídeo"
    )
    parser.add_argument("--fuente",  default="0",
                        help="Imagen (.png/.jpg), índice webcam (0,1…) o vídeo (.webm/.mp4) (def: 0)")
    parser.add_argument("--conf",    type=float, default=CONF_UMBRAL,
                        help=f"Confianza YOLO 0-1 (def: {CONF_UMBRAL})")
    parser.add_argument("--cada",    type=int,   default=CADA_N,
                        help=f"Analizar 1 de cada N frames en vídeo/webcam (def: {CADA_N})")
    parser.add_argument("--visual",  action="store_true",
                        help="Guardar imagen anotada en imgs/capturas/ (modo imagen)")
    parser.add_argument("--spots",   default=str(SPOTS_JSON),
                        help=f"Ruta al spots.json (def: {SPOTS_JSON})")
    args = parser.parse_args()

    print("▸ Cargando YOLOv8...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    print(f"▸ Cargando plazas desde '{args.spots}'...")
    spots = cargar_spots(Path(args.spots))
    print(f"  → {len(spots)} plazas cargadas\n")

    modo = detectar_modo(args.fuente)

    if modo == "imagen":
        modo_imagen(model, Path(args.fuente), spots, args.conf, args.visual)
    else:
        modo_video(model, args.fuente, spots, args.conf, args.cada)


if __name__ == "__main__":
    main()