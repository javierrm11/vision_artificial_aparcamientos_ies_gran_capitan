"""
marcar_plazas.py
────────────────────────────────────────────────
Marca polígonos de plazas de aparcamiento sobre
una imagen usando OpenCV (sin Colab ni ipywidgets).

Controles:
  Clic izquierdo  → añadir vértice
  Doble clic      → cerrar plaza (mín. 3 puntos)
  U               → deshacer (último punto o última plaza)
  C               → cancelar plaza en curso
  S               → guardar spots.json
  Q / ESC         → salir (pregunta si guardar)
────────────────────────────────────────────────
"""

import cv2
import json
import numpy as np
from pathlib import Path

# ── Configuración ────────────────────────────────────────────
IMG_PATH   = Path("imgs/1.png")          # imagen de referencia
SPOTS_JSON = Path("imgs/spots.json")     # donde se guardará el JSON
MAX_WIDTH  = 1280                        # ancho máximo de ventana
# ─────────────────────────────────────────────────────────────

# Colores BGR
COL_CLOSED  = (0, 220, 80)    # verde  — plazas cerradas
COL_CURRENT = (0, 220, 255)   # amarillo — polígono en curso
COL_POINT   = (0, 180, 255)   # naranja  — vértices en curso
COL_TEXT    = (0, 220, 80)
COL_STATUS  = (255, 255, 255)


def load_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen: {path}")
    return img


def compute_scale(img, max_width=MAX_WIDTH):
    h, w = img.shape[:2]
    scale = min(1.0, max_width / w)
    return scale


def resize(img, scale):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def render(base_img, spots, current_pts, scale, status_msg=""):
    """Dibuja plazas cerradas y el polígono en curso sobre base_img (escala display)."""
    canvas = resize(base_img, scale).copy()
    overlay = canvas.copy()

    # ── Plazas cerradas ──
    for s in spots:
        pts_sc = np.array(
            [[int(p[0] * scale), int(p[1] * scale)] for p in s["points"]],
            dtype=np.int32
        )
        cv2.fillPoly(overlay, [pts_sc], COL_CLOSED)
        cv2.polylines(canvas, [pts_sc], isClosed=True, color=COL_CLOSED, thickness=2)

        cx = int(pts_sc[:, 0].mean())
        cy = int(pts_sc[:, 1].mean())
        cv2.putText(canvas, str(s["id"]), (cx - 8, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_TEXT, 2)

    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    # ── Polígono en curso ──
    if current_pts:
        pts_sc = [(int(p[0] * scale), int(p[1] * scale)) for p in current_pts]
        for p in pts_sc:
            cv2.circle(canvas, p, 5, COL_POINT, -1)
        if len(pts_sc) > 1:
            cv2.polylines(canvas,
                          [np.array(pts_sc, dtype=np.int32)],
                          isClosed=False, color=COL_CURRENT, thickness=2)

    # ── Barra de estado ──
    bar_h = 36
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, h - bar_h), (w, h), (40, 40, 40), -1)
    cv2.putText(canvas, status_msg, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_STATUS, 1)

    # ── Leyenda ──
    legend = "[DblClic] cerrar  [U] deshacer  [C] cancelar  [S] guardar  [Q] salir"
    cv2.putText(canvas, legend, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return canvas


def save_spots(spots, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(spots, f, indent=2)
    print(f"✓ {len(spots)} plazas guardadas en '{path}'")


def main():
    base_img    = load_image(IMG_PATH)
    scale       = compute_scale(base_img)
    spots       = []
    current_pts = []
    status_msg  = f"Imagen cargada: {IMG_PATH}  |  Plazas: 0"

    WIN = "Marcador de plazas"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    last_click_time = [0]   # lista para mutabilidad en closure

    def mouse_cb(event, x, y, flags, param):
        nonlocal status_msg

        # Coordenadas en la imagen original
        ox = int(x / scale)
        oy = int(y / scale)

        if event == cv2.EVENT_LBUTTONDBLCLK:
            last_click_time[0] = 0   # resetear para no contar el dbl como clic simple
            if len(current_pts) >= 3:
                spot_id = len(spots)
                spots.append({"id": spot_id, "points": [list(p) for p in current_pts]})
                current_pts.clear()
                status_msg = f"✓ Plaza {spot_id} cerrada  |  Total: {len(spots)}"
            else:
                status_msg = "⚠ Necesitas al menos 3 puntos para cerrar una plaza"

        elif event == cv2.EVENT_LBUTTONDOWN:
            current_pts.append([ox, oy])
            status_msg = f"Punto {len(current_pts)} añadido ({ox}, {oy})  |  Plazas: {len(spots)}"

    cv2.setMouseCallback(WIN, mouse_cb)

    while True:
        frame = render(base_img, spots, current_pts, scale, status_msg)
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), ord('Q'), 27):   # Q o ESC — salir
            ans = input("¿Guardar antes de salir? [s/N]: ").strip().lower()
            if ans == 's':
                save_spots(spots, SPOTS_JSON)
            break

        elif key in (ord('s'), ord('S')):      # S — guardar
            save_spots(spots, SPOTS_JSON)
            status_msg = f"✓ Guardado  |  Plazas: {len(spots)}"

        elif key in (ord('u'), ord('U')):      # U — deshacer
            if current_pts:
                current_pts.pop()
                status_msg = "Último punto eliminado"
            elif spots:
                removed = spots.pop()
                status_msg = f"Plaza {removed['id']} eliminada"

        elif key in (ord('c'), ord('C')):      # C — cancelar plaza en curso
            current_pts.clear()
            status_msg = "Plaza en curso cancelada"

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()