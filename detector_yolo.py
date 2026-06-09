"""
================================================================================
MÓDULO CORE: MOTOR DE INFERENCIA SENSORIAL Y GEOMETRÍA COMPUTACIONAL
================================================================================
Asignatura: Visión Artificial | Proyecto: Gestión de Aparcamiento Inteligente
Desarrollado por: Javier y Pako
================================================================================
"""

import cv2
import json
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# REFACTORIZACIÓN INTEGRAL: Importación estricta de vuestros parámetros validados
from config import (
    SPOTS_JSON, OUTPUT_DIR, ESTADO_ACTUAL, NUM_PLAZAS, CONF_UMBRAL,
    SOLAPAMIENTO_MIN, MODELO_YOLO, COL_LIBRE, COL_OCUPADA, COL_YOLO
)


def cargar_spots(path: Path) -> list:
    """Deserializa de forma segura el archivo de configuración de plazas."""
    if not path.exists():
        print(f"⚠ Alerta: Archivo de plazas no encontrado en '{path}'")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def bbox_en_zona(x1, y1, x2, y2, puntos) -> bool:
    """
    GEOMETRÍA COMPUTACIONAL (IoU LOCAL):
    Calcula la fracción del Bounding Box de la IA contenida dentro de la ROI de la plaza.
    Aplica una operación matricial binaria de conteo de densidades distintas de cero.
    """
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    
    # Translación lineal al espacio de coordenadas local de la caja delimitadora
    pts_local = np.array([[p[0] - x1, p[1] - y1] for p in puntos], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Rellenamos la ROI local en la máscara de bits
    cv2.fillPoly(mask, [pts_local], 255)
    
    # Evaluación matemática de coincidencia contra vuestro umbral estricto SOLAPAMIENTO_MIN
    return np.count_nonzero(mask) / (w * h) >= SOLAPAMIENTO_MIN


def detectar(model, frame, conf) -> list:
    """Ejecuta la inferencia sobre el tensor de imagen y extrae las bounding boxes."""
    results = model(frame, conf=conf, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes.append((x1, y1, x2, y2))
    return boxes


def calcular_resultados(spots, boxes) -> list:
    """Evalúa de forma espacial cuántas entidades YOLO intersectan en cada ROI."""
    resultados = []
    for s in spots:
        coches = [b for b in boxes if bbox_en_zona(*b, s["points"])]
        resultados.append({
            "id":            s["id"],
            "coches_dentro": len(coches),
        })
    return resultados


def dibujar(frame, spots, resultados, boxes) -> np.ndarray:
    """Capa de visualización: Genera la superposición de colores semáforo y cajas YOLO."""
    overlay = frame.copy()

    total_coches = sum(r["coches_dentro"] for r in resultados)
    libres       = max(0, NUM_PLAZAS - total_coches)
    ocupadas     = NUM_PLAZAS - libres

    # Pintado de las Regiones de Interés Poligonales
    for s, r in zip(spots, resultados):
        # Lógica semáforo: Si la zona específica contiene vehículos, cambia de estado
        color  = COL_OCUPADA if r["coches_dentro"] > 0 else COL_LIBRE
        pts_np = np.array(s["points"], dtype=np.int32)
        
        cv2.fillPoly(overlay, [pts_np], color)
        cv2.polylines(frame, [pts_np], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        
        cx = int(pts_np[:, 0].mean())
        cy = int(pts_np[:, 1].mean())
        cv2.putText(frame, f"Z{s['id']}: {r['coches_dentro']} veh",
                    (cx - 35, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)

    # Fusión alfanumérica de transparencia (Blending)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # Renderizado de los Bounding Boxes de YOLO (Color Naranja Corporativo)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_YOLO, 1, lineType=cv2.LINE_AA)
        cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 4, (0, 0, 255), -1)

    # Panel de control de usuario (HUD Superior)
    cv2.rectangle(frame, (0, 0), (420, 35), (30, 30, 30), -1)
    cv2.putText(frame,
                f"Libres: {libres}/{NUM_PLAZAS} | Ocupadas: {ocupadas}/{NUM_PLAZAS}",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def dibujar_leyenda_video(frame) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "[G] Guardar Captura   [Q/ESC] Salir",
                (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def guardar_estado_actual(resultados):
    """Vuelca el estado transaccional numérico en un archivo JSON plano."""
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
    with open(ESTADO_ACTUAL, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)


def guardar_captura(resultados, fuente, frame=None):
    """Almacena auditorías históricas multimedia en el disco."""
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
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)

    if frame is not None:
        img_path = OUTPUT_DIR / f"captura_{ts}.png"
        cv2.imwrite(str(img_path), frame)
        print(f"✓ Guardado: {img_path.name} y {json_path.name}")
    else:
        print(f"✓ JSON guardado: {json_path.name}")


def modo_imagen(model, path: Path, spots, conf, guardar_visual):
    print(f"▸ Modo IMAGEN: {path}")
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"No se encontró el recurso: {path}")

    boxes = detectar(model, frame, conf)
    print(f"  → {len(boxes)} objetos detectados")

    resultados = calcular_resultados(spots, boxes)
    frame_viz  = dibujar(frame.copy(), spots, resultados, boxes)

    guardar_estado_actual(resultados)
    guardar_captura(resultados, path, frame=frame_viz if guardar_visual else None)

    cv2.imshow("Detector aparcamiento — Modo Estatico", frame_viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def modo_video(model, fuente, spots, conf, cada):
    src  = int(fuente) if str(fuente).isdigit() else str(fuente)
    tipo = "WEBCAM" if isinstance(src, int) else f"VÍDEO ({fuente})"
    print(f"▸ Modo {tipo}")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo inicializar la fuente: {fuente}")

    resultados  = [{"id": s["id"], "coches_dentro": 0} for s in spots]
    boxes       = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Bucle continuo si es un archivo de vídeo fijo
            if isinstance(src, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_count += 1

        # Optimización computacional: Inferencia diferida 1 de cada N fotogramas
        if frame_count % cada == 0:
            boxes      = detectar(model, frame, conf)
            resultados = calcular_resultados(spots, boxes)

        frame_viz = dibujar(frame.copy(), spots, resultados, boxes)
        frame_viz = dibujar_leyenda_video(frame_viz)
        cv2.imshow("Detector de Aparcamiento Inteligente", frame_viz)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('g'), ord('G')):
            guardar_captura(resultados, fuente, frame=frame_viz)

    cap.release()
    cv2.destroyAllWindows()


EXTENSIONES_IMAGEN = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
EXTENSIONES_VIDEO  = {".webm", ".mp4", ".avi", ".mov", ".mkv"}


def detectar_modo(fuente: str) -> str:
    sufijo = Path(fuente).suffix.lower()
    if sufijo in EXTENSIONES_IMAGEN:
        return "imagen"
    return "video"


def main():
    parser = argparse.ArgumentParser(description="Detector de aparcamiento — pipeline core")
    parser.add_argument("--fuente",  default="0", help="Recurso multimedia o ID cámara web")
    parser.add_argument("--conf",    type=float, default=CONF_UMBRAL, help="Umbral analítico")
    parser.add_argument("--cada",    type=int, default=5, help="Muestreo adaptativo de frames")
    parser.add_argument("--visual",  action="store_true", help="Salida gráfica en disco")
    parser.add_argument("--spots",   default=str(SPOTS_JSON), help="Ruta al mapa JSON")
    args = parser.parse_args()

    print(f"▸ Inicializando Framework de IA: {MODELO_YOLO}...")
    from ultralytics import YOLO
    # Aquí instanciamos de verdad el modelo matemático con la clase YOLO
    model_instanciado = YOLO(MODELO_YOLO)

    spots = cargar_spots(Path(args.spots))
    print(f"  → {len(spots)} zonas cargadas en la caché estructural\n")

    modo = detectar_modo(args.fuente)

    if modo == "imagen":
        modo_imagen(model_instanciado, Path(args.fuente), spots, args.conf, args.visual)
    else:
        modo_video(model_instanciado, args.fuente, spots, args.conf, args.cada)


if __name__ == "__main__":
    main()