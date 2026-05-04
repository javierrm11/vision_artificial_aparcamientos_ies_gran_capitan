"""
detector.py
────────────────────────────────────────────────
Detecta plazas libres/ocupadas usando varianza
de píxeles dentro de cada polígono.

Uso:
    python detector.py --img imgs/foto.png
    python detector.py --img imgs/foto.png --umbral 800

Salida:
    imgs/estado.json   ← resultado por plaza
    imgs/resultado.png ← imagen anotada (opcional con --visual)
────────────────────────────────────────────────
"""

import cv2
import json
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# ── Configuración por defecto ─────────────────
SPOTS_JSON   = Path("imgs/spots.json")
OUTPUT_JSON  = Path("imgs/estado.json")
OUTPUT_IMG   = Path("imgs/resultado.png")
UMBRAL_VAR   = 600   # por encima → OCUPADA (ajustar según tu cámara/luz)
# ─────────────────────────────────────────────

COL_LIBRE    = (0, 220, 80)    # verde
COL_OCUPADA  = (0, 60, 220)    # rojo


def cargar_spots(path: Path):
    with open(path) as f:
        return json.load(f)


def varianza_plaza(img_gray, puntos):
    """Calcula la varianza de los píxeles dentro del polígono."""
    mask = np.zeros(img_gray.shape[:2], dtype=np.uint8)
    pts  = np.array(puntos, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    pixeles = img_gray[mask == 255]
    if len(pixeles) == 0:
        return 0.0
    return float(np.var(pixeles))


def analizar(img_path: Path, spots, umbral: float, guardar_visual: bool):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"No se encontró: {img_path}")

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canvas  = img.copy()
    overlay = img.copy()

    resultados = []

    for s in spots:
        var    = varianza_plaza(gray, s["points"])
        estado = "OCUPADA" if var > umbral else "LIBRE"
        color  = COL_OCUPADA if estado == "OCUPADA" else COL_LIBRE

        resultados.append({
            "id":       s["id"],
            "estado":   estado,
            "varianza": round(var, 1)
        })

        # Dibujar en la imagen
        pts_np = np.array(s["points"], dtype=np.int32)
        cv2.fillPoly(overlay, [pts_np], color)
        cv2.polylines(canvas, [pts_np], isClosed=True, color=color, thickness=2)

        cx = int(pts_np[:, 0].mean())
        cy = int(pts_np[:, 1].mean())
        cv2.putText(canvas, f"{s['id']}:{estado[0]}",
                    (cx - 16, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    if guardar_visual:
        cv2.imwrite(str(OUTPUT_IMG), canvas)
        print(f"✓ Imagen anotada guardada en '{OUTPUT_IMG}'")

    return resultados, canvas


def guardar_json(resultados, img_path: Path):
    libres   = [r for r in resultados if r["estado"] == "LIBRE"]
    ocupadas = [r for r in resultados if r["estado"] == "OCUPADA"]

    salida = {
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "imagen":         str(img_path),
        "total_plazas":   len(resultados),
        "libres":         len(libres),
        "ocupadas":       len(ocupadas),
        "plazas":         resultados
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    print(f"✓ Estado guardado en '{OUTPUT_JSON}'")
    print(f"  Total: {len(resultados)}  |  Libres: {len(libres)}  |  Ocupadas: {len(ocupadas)}")
    for r in resultados:
        icono = "🟢" if r["estado"] == "LIBRE" else "🔴"
        print(f"  {icono} Plaza {r['id']:>2}  {r['estado']:<8}  var={r['varianza']}")


def main():
    parser = argparse.ArgumentParser(description="Detector de plazas de aparcamiento")
    parser.add_argument("--img",     required=True,            help="Imagen a analizar")
    parser.add_argument("--umbral",  type=float, default=UMBRAL_VAR, help="Umbral de varianza (def: 600)")
    parser.add_argument("--visual",  action="store_true",      help="Guardar imagen anotada")
    parser.add_argument("--spots",   default=str(SPOTS_JSON),  help="Ruta al spots.json")
    args = parser.parse_args()

    spots      = cargar_spots(Path(args.spots))
    resultados, _ = analizar(Path(args.img), spots, args.umbral, args.visual)
    guardar_json(resultados, Path(args.img))


if __name__ == "__main__":
    main()