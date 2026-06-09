"""
================================================================================
MÓDULO DE CALIBRACIÓN GEOMÉTRICA: CALIBRADOR DE REGIONES DE INTERÉS (ROI)
================================================================================
Asignatura: Visión Artificial | Proyecto: Gestión de Aparcamiento Inteligente
Desarrollado por: Javier y Pako 

PROPÓSITO TÉCNICO:
Este script resuelve un problema de diseño crítico. Los modelos de Deep Learning
como YOLO son óptimos detectando entidades físicas (coches), pero sufren tasas
críticas de falsos positivos al intentar detectar "la nada" (plazas vacías). 
Este módulo actúa como una etapa de calibración estática que digitaliza la 
geometría del parking mediante OpenCV, abstrayendo las zonas de estacionamiento
en estructuras de datos poligonales (arrays de NumPy) indexadas en un archivo JSON.
================================================================================
"""

import cv2
import json
import numpy as np
from pathlib import Path

# REFACTORIZACIÓN INTEGRAL: Acoplamiento seguro a la configuración centralizada
from config import SPOTS_JSON, COL_LIBRE, COL_OCUPADA

# Configuraciones de interfaz locales (Mantenemos vuestras preferencias estéticas)
IMG_PATH    = Path("imgs/1.png")          # Frame estático base del parking
MAX_WIDTH   = 1280                        # Factor de escala límite de pantalla
COL_CLOSED  = COL_LIBRE                   # Verde oficial de las plazas disponibles
COL_CURRENT = (0, 220, 255)               # Amarillo — Aristas en fase de dibujo
COL_POINT   = (0, 180, 255)               # Naranja — Vértices registrados
COL_STATUS  = (255, 255, 255)             # Blanco — Texto HUD


def load_spots(json_path: Path) -> list:
    """Carga desde disco el histórico de plazas calibradas previamente."""
    if not json_path.exists():
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ Error al leer {json_path}: {e}")
        return []


def save_spots(spots: list, json_path: Path):
    """Vuelca las coordenadas estructurales a un fichero estructurado JSON."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(spots, f, indent=2)


def render(base_img, spots, current_pts, scale, status_msg) -> np.ndarray:
    """Motor de renderizado gráfico OpenCV en tiempo real sobre búfer temporal."""
    frame = base_img.copy()

    # 1. RENDERIZADO DE PLAZAS CONSOLIDADAS
    for spot in spots:
        spot_id = spot["id"]
        pts = np.array(spot["points"], np.int32)
        pts_scaled = (pts * scale).astype(np.int32)
        
        cv2.polylines(frame, [pts_scaled], isClosed=True, color=COL_CLOSED, thickness=2)
        
        moments = cv2.moments(pts_scaled)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = pts_scaled[0][0], pts_scaled[0][1]
            
        cv2.putText(frame, str(spot_id), (cx - 8, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_CLOSED, 2, cv2.LINE_AA)

    # 2. RENDERIZADO DEL POLÍGONO EN CONSTRUCCIÓN (Línea elástica)
    if current_pts:
        pts_curr = (np.array(current_pts, np.float32) * scale).astype(np.int32)
        for pt in pts_curr:
            cv2.circle(frame, tuple(pt), 4, COL_POINT, -1, cv2.LINE_AA)
        if len(pts_curr) > 1:
            cv2.polylines(frame, [pts_curr], isClosed=False, color=COL_CURRENT, thickness=2)

    # 3. INTERFAZ DE USUARIO (HUD)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 35), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, status_msg, (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_STATUS, 1, cv2.LINE_AA)

    return frame


def main():
    if not IMG_PATH.exists():
        print(f"❌ Error crítico: No se encuentra la imagen de referencia en {IMG_PATH}")
        return

    orig_img = cv2.imread(str(IMG_PATH))
    orig_h, orig_w = orig_img.shape[:2]

    scale = 1.0
    if orig_w > MAX_WIDTH:
        scale = MAX_WIDTH / orig_w
        
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    base_img = cv2.resize(orig_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    spots = load_spots(SPOTS_JSON)
    current_pts = []
    pendiente_capacidad = None  # ID de zona recién cerrada que espera input de capacidad
    status_msg = f"Controles — Clic: Añadir | Doble Clic: Cerrar Plaza | Plazas: {len(spots)}"

    WIN = "Calibrador de Plazas de Aparcamiento"
    cv2.namedWindow(WIN)

    def mouse_cb(event, x, y, flags, param):
        nonlocal current_pts, status_msg, spots, pendiente_capacidad
        ox = int(x / scale)
        oy = int(y / scale)

        if event == cv2.EVENT_LBUTTONDBLCLK:
            if len(current_pts) >= 2:
                current_pts.pop()
                current_pts.pop()

            if len(current_pts) >= 3:
                next_id = max([s["id"] for s in spots]) + 1 if spots else 0
                spots.append({"id": next_id, "points": list(current_pts)})
                current_pts = []
                pendiente_capacidad = next_id
                status_msg = f"✓ Zona {next_id} cerrada — escribe la capacidad en la consola"
            else:
                status_msg = "⚠ Error geométrico: Se requieren mínimo 3 vértices para cerrar una ROI"

        elif event == cv2.EVENT_LBUTTONDOWN:
            current_pts.append([ox, oy])
            status_msg = f"Vértice {len(current_pts)} registrado ({ox}, {oy}) | Plazas: {len(spots)}"

    cv2.setMouseCallback(WIN, mouse_cb)

    while True:
        frame = render(base_img, spots, current_pts, scale, status_msg)
        cv2.imshow(WIN, frame)

        # Preguntar capacidad real en consola justo tras cerrar una zona
        if pendiente_capacidad is not None:
            cv2.waitKey(1)  # forzar refresco antes de bloquear en input()
            zone_id = pendiente_capacidad
            pendiente_capacidad = None
            try:
                cap_str = input(f"  ¿Cuántas plazas tiene la Zona {zone_id}? [Enter = automático]: ").strip()
                if cap_str.isdigit() and int(cap_str) > 0:
                    spots[-1]["capacidad"] = int(cap_str)
                    status_msg = f"✓ Zona {zone_id}: {cap_str} plazas reales | Total zonas: {len(spots)}"
                else:
                    status_msg = f"✓ Zona {zone_id} cerrada (capacidad automática) | Total: {len(spots)}"
            except (EOFError, ValueError):
                status_msg = f"✓ Zona {zone_id} cerrada (capacidad automática) | Total: {len(spots)}"

        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            ans = input("⚠ Se han detectado cambios. ¿Desea salvar la sesión? [s/N]: ").strip().lower()
            if ans == 's':
                save_spots(spots, SPOTS_JSON)
            break
        elif key in (ord('s'), ord('S')):
            save_spots(spots, SPOTS_JSON)
            status_msg = f"✓ Archivo '{SPOTS_JSON}' serializado correctamente | Total: {len(spots)}"
        elif key in (ord('u'), ord('U')):
            if current_pts:
                current_pts.pop()
                status_msg = "Deshacer: Último vértice eliminado"
            elif spots:
                spots.pop()
                status_msg = "Deshacer: Última plaza consolidada eliminada"
        elif key in (ord('c'), ord('C')):
            current_pts = []
            status_msg = "Operación cancelada: Buffer de vértices vaciado"

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()