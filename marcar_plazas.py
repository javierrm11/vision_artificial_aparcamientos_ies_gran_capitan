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

# ── CONFIGURACIÓN DE RUTAS Y LIMITACIONES DE ENTORNO ──────────────────────────
# Encapsulamos las rutas utilizando la librería orientada a objetos 'pathlib'.
IMG_PATH   = Path("imgs/1.png")          # Matriz de referencia visual fija (webcam)
SPOTS_JSON = Path("imgs/spots.json")     # Repositorio de persistencia de las ROIs
MAX_WIDTH  = 1280                        # Factor de escala límite para evitar desbordamiento en monitores pequeños
# ─────────────────────────────────────────────────────────────────────────────

# PALETA CROMÁTICA EN FORMATO BGR (OpenCV Estándar)
# OpenCV trabaja nativamente en BGR. Definimos los canales de color.
COL_CLOSED  = (0, 220, 80)    # Verde  — Polígonos cerrados (Estructura ROI consolidada)
COL_CURRENT = (0, 220, 255)   # Amarillo — Aristas del polígono en fase de dibujo
COL_POINT   = (0, 180, 255)   # Naranja  — Vértices (puntos de control dimensional)
COL_TEXT    = (0, 220, 80)    # Verde  — Texto de información general
COL_STATUS  = (255, 255, 255) # Blanco — Barra de estado dinámica (Feedback en tiempo real)


def load_spots(json_path: Path) -> list:
    """
    DESERIALIZACIÓN DE COORDENADAS:
    Carga desde disco el histórico de plazas calibradas previamente. 
    Transforma listas planas JSON de vuelta a tipos nativos de Python.
    """
    if not json_path.exists():
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ Error al leer {json_path}: {e}")
        return []


def save_spots(spots: list, json_path: Path):
    """
    SERIALIZACIÓN Y PERSISTENCIA DE DATOS:
    Vuelca las coordenadas estructurales a un fichero estructurado JSON.
    Garantiza que el script 'detector_yolo.py' pueda consumir los datos de forma asíncrona.
    """
    # Creamos de forma recursiva el directorio padre si no existiese previamente
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        # indent=2 optimiza la legibilidad del archivo para auditorías o modificaciones manuales
        json.dump(spots, f, indent=2)


def render(base_img, spots, current_pts, scale, status_msg) -> np.ndarray:
    """
    MOTOR DE RENDERIZADO GRÁFICO EN TIEMPO REAL:
    Toma la imagen base y, aplicando primitivas gráficas de OpenCV sobre una copia de la matriz,
    dibuja el estado actual de la calibración sin alterar el archivo original.
    """
    # .copy() es mandatorio para limpiar el buffer de dibujo en cada ciclo del loop (Evita el efecto arrastre)
    frame = base_img.copy()

    # 1. RENDERIZADO DE PLAZAS CONSOLIDADAS (Historial)
    for spot in spots:
        spot_id = spot["id"]
        pts = np.array(spot["points"], np.int32)
        
        # Redimensionamos las coordenadas escaladas al tamaño actual de la pantalla de visualización
        pts_scaled = (pts * scale).astype(np.int32)
        
        # cv2.polylines: Traza un polígono cerrado (isClosed=True) uniendo la secuencia de vértices
        cv2.polylines(frame, [pts_scaled], isClosed=True, color=COL_CLOSED, thickness=2)
        
        # Cálculo del centroide del polígono para posicionar el ID flotante en el espacio correcto
        moments = cv2.moments(pts_scaled)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            # Plan B matemático si el cálculo por momentos colapsa (líneas concurrentes)
            cx, cy = pts_scaled[0][0], pts_scaled[0][1]
            
        cv2.putText(frame, str(spot_id), (cx - 8, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_CLOSED, 2, cv2.LINE_AA)

    # 2. RENDERIZADO DEL POLÍGONO EN CONSTRUCCIÓN (Fase activa)
    if current_pts:
        pts_curr = (np.array(current_pts, np.float32) * scale).astype(np.int32)
        
        # Pintamos los vértices individuales como círculos discretos (radio 4 píxeles)
        for pt in pts_curr:
            cv2.circle(frame, tuple(pt), 4, COL_POINT, -1, cv2.LINE_AA)
            
        # Si el usuario ha trazado al menos 2 puntos, mostramos la línea de tensión elástica
        if len(pts_curr) > 1:
            cv2.polylines(frame, [pts_curr], isClosed=False, color=COL_CURRENT, thickness=2)

    # 3. INTERFAZ DE USUARIO INTEGRADA (HUD / OSD)
    # Dibujamos un rectángulo inferior opaco que actúa como contenedor de la barra de estado
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 35), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, status_msg, (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_STATUS, 1, cv2.LINE_AA)

    return frame


def main():
    # Validación crítica: Detener ejecución si la fuente de datos (imagen de la webcam) no existe
    if not IMG_PATH.exists():
        print(f"Error crítico: No se encuentra la imagen de referencia en {IMG_PATH}")
        return

    # Lectura de la imagen en formato BGR (Nativo de OpenCV)
    orig_img = cv2.imread(str(IMG_PATH))
    orig_h, orig_w = orig_img.shape[:2]

    # CÁLCULO DILATACIÓN DE ESCALA: Garantiza la responsividad en pantallas de portátiles
    scale = 1.0
    if orig_w > MAX_WIDTH:
        scale = MAX_WIDTH / orig_w
        
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # Redimensionamos la matriz de visualización utilizando interpolación de área (Óptima para reducciones)
    base_img = cv2.resize(orig_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Inicialización de estructuras en memoria
    spots = load_spots(SPOTS_JSON)
    current_pts = [] # Acumulador de vértices temporales
    status_msg = f"Controles — Clic: Añadir | Doble Clic: Cerrar Plaza | Plazas: {len(spots)}"

    WIN = "Calibrador de Plazas de Aparcamiento"
    cv2.namedWindow(WIN)

    # ── PROGRAMACIÓN ASÍNCRONA BASADA EN EVENTOS DE RATÓN (CALLBACKS) ─────────
    # No usamos un bucle síncrono para leer el ratón. El sistema operativo 
    # interrumpe la ejecución e invoca esta función ('mouse_cb') de 
    # manera asíncrona cuando ocurre un evento.
    def mouse_cb(event, x, y, flags, param):
        nonlocal current_pts, status_msg, spots

        # Conversión espacial inversa: Llevamos las coordenadas de la ventana (escalada)
        # de vuelta a las dimensiones reales de la imagen original (Matriz nativa).
        ox = int(x / scale)
        oy = int(y / scale)

        # EVENTO: DOBLE CLIC IZQUIERDO -> Consolidación y cierre de geometría
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # Eliminamos los dos falsos clics que Windows genera antes de registrar un doble clic nativo
            if len(current_pts) >= 2:
                current_pts.pop()
                current_pts.pop()

            # Validación geométrica elemental: Un polígono cerrado bidimensional requiere mínimo 3 vértices
            if len(current_pts) >= 3:
                # Determinamos el ID secuencial autoincremental para la plaza
                next_id = 0
                if spots:
                    next_id = max(s["id"] for s in spots) + 1
                
                # Almacenamos la estructura en memoria
                spots.append({
                    "id": next_id,
                    "points": list(current_pts)
                })
                current_pts = [] # Vaciamos el acumulador temporal
                status_msg = f"✓ Plaza {next_id} cerrada con éxito | Total: {len(spots)}"
            else:
                status_msg = "⚠ Error geométrico: Se requieren mínimo 3 vértices para cerrar una ROI"

        # EVENTO: CLIC IZQUIERDO SIMPLE -> Adición de vértice vectorizado
        elif event == cv2.EVENT_LBUTTONDOWN:
            current_pts.append([ox, oy])
            status_msg = f"Vértice {len(current_pts)} registrado ({ox}, {oy}) | Plazas: {len(spots)}"

    # Vinculamos la función callback a la ventana activa de OpenCV
    cv2.setMouseCallback(WIN, mouse_cb)

    # ── BUCLE DE RENDERIZADO PRINCIPAL (MAIN LOOP) ────────────────────────────
    while True:
        # 1. Actualizamos el frame gráfico procesando las listas dinámicas
        frame = render(base_img, spots, current_pts, scale, status_msg)
        cv2.imshow(WIN, frame)

        # 2. Captura síncrona del teclado mediante enmascaramiento binario (& 0xFF)
        key = cv2.waitKey(20) & 0xFF

        # CONTROL: Q o ESC -> Salida segura con flujo interactivo en terminal
        if key in (ord('q'), ord('Q'), 27):
            ans = input("⚠ Se han detectado cambios. ¿Desea salvar la sesión antes de salir? [s/N]: ").strip().lower()
            if ans == 's':
                save_spots(spots, SPOTS_JSON)
            break

        # CONTROL: S -> Forzar guardado inmediato en caliente
        elif key in (ord('s'), ord('S')):
            save_spots(spots, SPOTS_JSON)
            status_msg = f"✓ Archivo '{SPOTS_JSON}' serializado correctamente | Total: {len(spots)}"

        # CONTROL: U -> Función Deshacer (Undo) basada en estructuras de Pilas (LIFO)
        elif key in (ord('u'), ord('U')):
            if current_pts:
                current_pts.pop() # Elimina el último punto en construcción
                status_msg = "Deshacer: Último vértice en curso eliminado"
            elif spots:
                spots.pop() # Elimina la última plaza completamente consolidada
                status_msg = "Deshacer: Última plaza consolidada eliminada"
            else:
                status_msg = "⚠ Nada que deshacer en el histórico"

        # CONTROL: C -> Cancelar construcción activa
        elif key in (ord('c'), ord('C')):
            current_pts = []
            status_msg = "Operación cancelada: Buffer de vértices vaciado"

    # Liberación obligatoria de punteros gráficos del sistema operativo
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()