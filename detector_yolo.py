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
SPOTS_JSON     = Path("imgs/spots.json")
OUTPUT_DIR     = Path("imgs/capturas")
ESTADO_ACTUAL  = Path("imgs/estado_actual.json")
NUM_PLAZAS     = 22
CONF_UMBRAL    = 0.30
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


def bbox_en_zona(x1, y1, x2, y2, puntos):
    """True si algún punto del bbox o del polígono se solapan."""
    pts = np.array(puntos, dtype=np.float32)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    for px, py in [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(cx,cy)]:
        if cv2.pointPolygonTest(pts, (float(px), float(py)), False) >= 0:
            return True
    for px, py in puntos:
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True
    return False


def detectar(model, frame, conf):
    """Ejecuta YOLO sobre un frame y devuelve bboxes de todos los objetos."""
    results = model(frame, conf=conf, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes.append((x1, y1, x2, y2))
    return boxes


def calcular_resultados(spots, boxes):
    """Cuenta objetos detectados que solapan con cada zona ROI."""
    resultados = []
    for s in spots:
        coches = [b for b in boxes if bbox_en_zona(*b, s["points"])]
        resultados.append({
            "id":            s["id"],
            "coches_dentro": len(coches),
        })
    return resultados


def dibujar(frame, spots, resultados, boxes):
    """Dibuja zonas ROI, bboxes YOLO y HUD sobre el frame."""
    overlay = frame.copy()

    total_coches = sum(r["coches_dentro"] for r in resultados)
    libres       = max(0, NUM_PLAZAS - total_coches)
    ocupadas     = NUM_PLAZAS - libres

    for s, r in zip(spots, resultados):
        color  = COL_LIBRE if libres > 0 else COL_OCUPADA
        pts_np = np.array(s["points"], dtype=np.int32)
        cv2.fillPoly(overlay, [pts_np], color)
        cv2.polylines(frame, [pts_np], isClosed=True, color=color, thickness=2)
        cx = int(pts_np[:, 0].mean())
        cy = int(pts_np[:, 1].mean())
        cv2.putText(frame, f"Zona {s['id']}: {r['coches_dentro']} coches",
                    (cx - 50, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_YOLO, 1)
        cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 5, (0, 0, 255), -1)

    # HUD superior
    cv2.rectangle(frame, (0, 0), (400, 34), (30, 30, 30), -1)
    cv2.putText(frame,
                f"Libres: {libres}/{NUM_PLAZAS}  Ocupadas: {ocupadas}/{NUM_PLAZAS}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return frame


def dibujar_leyenda_video(frame):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "[G] guardar captura   [Q/ESC] salir",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def guardar_estado_actual(resultados):
    """Sobreescribe imgs/estado_actual.json con el estado más reciente."""
    total_coches = sum(r["coches_dentro"] for r in resultados)
    libres       = max(0, NUM_PLAZAS - total_coches)
    datos = {
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "total_plazas": NUM_PLAZAS,
        "libres":       libres,
        "ocupadas":     NUM_PLAZAS - libres,
        "zonas":        resultados
    }
    ESTADO_ACTUAL.parent.mkdir(parents=True, exist_ok=True)
    with open(ESTADO_ACTUAL, "w") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)


def guardar_captura(resultados, fuente, frame=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    total_coches = sum(r["coches_dentro"] for r in resultados)
    libres       = max(0, NUM_PLAZAS - total_coches)
    datos = {
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "fuente":       str(fuente),
        "total_plazas": NUM_PLAZAS,
        "libres":       libres,
        "ocupadas":     NUM_PLAZAS - libres,
        "zonas":        resultados
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

    print(f"  🟢 Libres: {libres}  🔴 Ocupadas: {NUM_PLAZAS - libres}")
    for r in resultados:
        print(f"  Zona {r['id']:>2}  Coches dentro: {r['coches_dentro']}")


# ══════════════════════════════════════════════
# Modos de ejecución
# ══════════════════════════════════════════════

def modo_imagen(model, path: Path, spots, conf, guardar_visual):
    print(f"▸ Modo IMAGEN: {path}")
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"No se encontró: {path}")

    boxes = detectar(model, frame, conf)
    print(f"  → {len(boxes)} objetos detectados")

    resultados = calcular_resultados(spots, boxes)
    frame_viz  = dibujar(frame.copy(), spots, resultados, boxes)

    guardar_estado_actual(resultados)
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
            boxes      = detectar(model, frame, conf)
            resultados = calcular_resultados(spots, boxes)

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
    parser.add_argument("--visual",  action="store_true",
                        help="Guardar imagen anotada en imgs/capturas/ (modo imagen)")
    parser.add_argument("--spots",   default=str(SPOTS_JSON),
                        help=f"Ruta al spots.json (def: {SPOTS_JSON})")
    args = parser.parse_args()

    print("▸ Cargando YOLOv8...")
    from ultralytics import YOLO
    model = YOLO("yolo11x.pt")

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