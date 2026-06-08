# Detector de Aparcamiento con Visión Artificial

Sistema de detección de plazas libres/ocupadas en aparcamientos mediante cámara de vigilancia y YOLOv8.

---

## Cómo funciona

El sistema se compone de dos herramientas independientes:

```
marcar_plazas.py   →   spots.json   →   detector_yolo.py
   (dibujar ROIs)      (zonas)         (detectar + contar)
```

1. **`marcar_plazas.py`**: se ejecuta una sola vez para dibujar polígonos (zonas ROI) sobre una captura del aparcamiento. Guarda las zonas en `imgs/spots.json`.
2. **`detector_yolo.py`**: carga las zonas y analiza en tiempo real (webcam, vídeo o imagen) cuántos objetos hay dentro de cada zona. Calcula `Libres = NUM_PLAZAS − coches_detectados`.

La detección usa **YOLOv8n** (nano) para máxima velocidad. Cualquier objeto que solape con una zona ROI cuenta como plaza ocupada — no solo el centro del bbox, sino cualquier punto del rectángulo de detección.

---

## Requisitos

- Python 3.8+
- Entorno virtual recomendado

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

```bash
pip install ultralytics opencv-python numpy
```

El modelo `yolov8n.pt` se descarga automáticamente la primera vez que se ejecuta el detector.

---

## Estructura de archivos

```
aparcamiento/
├── marcar_plazas.py      # Herramienta para marcar zonas ROI
├── detector_yolo.py      # Detector principal
├── yolov8n.pt            # Modelo YOLOv8 nano (se descarga automáticamente)
└── imgs/
    ├── 1.png             # Imagen de referencia para marcar zonas
    ├── spots.json        # Zonas ROI guardadas
    └── capturas/         # Capturas guardadas con G (se genera automáticamente)
        ├── captura_YYYYMMDD_HHMMSS.png
        └── estado_YYYYMMDD_HHMMSS.json
```

---

## Paso 1 — Marcar las zonas ROI

Coloca una captura del aparcamiento en `imgs/1.png` y ejecuta:

```bash
python marcar_plazas.py
```

### Controles

| Acción | Resultado |
|--------|-----------|
| Clic izquierdo | Añadir vértice al polígono |
| Doble clic | Cerrar zona (mínimo 3 puntos) |
| `U` | Deshacer último punto o última zona |
| `C` | Cancelar zona en curso |
| `S` | Guardar `spots.json` |
| `Q` / `ESC` | Salir (pregunta si guardar) |

Cada zona marcada representa un **grupo de plazas** (pasillo izquierdo, pasillo derecho, etc.). El número total de plazas del aparcamiento se configura con `NUM_PLAZAS` en `detector_yolo.py`.

---

## Paso 2 — Ejecutar el detector

### Imagen estática

```bash
python detector_yolo.py --fuente imgs/foto.png
```

### Archivo de vídeo

```bash
python detector_yolo.py --fuente imgs/video.mp4
python detector_yolo.py --fuente imgs/video.webm
python detector_yolo.py --fuente imgs/video.avi
```

### Webcam

```bash
python detector_yolo.py                  # webcam por defecto (índice 0)
python detector_yolo.py --fuente 1       # segunda webcam
```

### Opciones

| Argumento | Por defecto | Descripción |
|-----------|-------------|-------------|
| `--fuente` | `0` | Imagen, vídeo o índice de webcam |
| `--conf` | `0.35` | Confianza mínima YOLO (0.0 – 1.0) |
| `--cada` | `5` | Analizar 1 de cada N frames (vídeo/webcam) |
| `--visual` | off | Guardar imagen anotada al analizar imagen estática |
| `--spots` | `imgs/spots.json` | Ruta al archivo de zonas |

### Controles en vídeo/webcam

| Tecla | Acción |
|-------|--------|
| `G` | Guardar captura actual (imagen PNG + JSON de estado) |
| `Q` / `ESC` | Salir |

---

## Configuración principal (`detector_yolo.py`)

```python
NUM_PLAZAS  = 22      # Total de plazas del aparcamiento
CONF_UMBRAL = 0.35    # Confianza mínima YOLO
CADA_N      = 5       # Analizar 1 de cada N frames
```

Ajusta `NUM_PLAZAS` al número real de plazas de tu aparcamiento. El sistema calcula:

```
Libres   = NUM_PLAZAS − coches detectados en zonas ROI
Ocupadas = NUM_PLAZAS − Libres
```

---

## Formato de `spots.json`

```json
[
  {
    "id": 0,
    "points": [[120, 80], [450, 75], [480, 420], [100, 430]]
  },
  {
    "id": 1,
    "points": [[820, 60], [1100, 55], [1120, 380], [840, 390]]
  }
]
```

Cada entrada define un polígono con sus vértices en coordenadas de la imagen original (sin escalar).

---

## Formato del JSON de captura

Al pulsar `G` o al analizar una imagen con `--visual`, se genera:

```json
{
  "timestamp": "2026-03-17T17:42:00",
  "fuente": "imgs/video.mp4",
  "total_plazas": 22,
  "libres": 9,
  "ocupadas": 13,
  "zonas": [
    { "id": 0, "coches_dentro": 5 },
    { "id": 1, "coches_dentro": 4 },
    { "id": 2, "coches_dentro": 4 }
  ]
}
```

---

## Mejoras posibles

| Mejora | Descripción |
|--------|-------------|
| **Suavizado temporal** | Marcar una plaza ocupada solo si lleva N frames consecutivos con coche, evitando falsos positivos por coches en movimiento |
| **Umbral de solapamiento** | Exigir que el X% del bbox esté dentro de la zona, no solo un píxel de contacto |
| **Filtro por clase** | Ignorar personas/bicicletas que crucen la zona usando las clases COCO (2=car, 3=moto, 5=bus, 7=truck) |
| **Zona de exclusión/pasillo** | Dibujar zonas que se excluyen del conteo para evitar contar coches en movimiento |
| **Histórico de ocupación** | Registrar el estado cada N minutos en CSV/SQLite para graficar la ocupación a lo largo del día |
| **API REST** | Exponer el estado actual en JSON vía HTTP (`/api/estado`) para integrarlo con una web o panel |
| **Alertas** | Notificar por email/webhook cuando el aparcamiento se llena o queda libre |
| **Modelo más preciso** | Cambiar `yolov8n.pt` por `yolov8s.pt` o `yolov8m.pt` para mejor detección a costa de más CPU |

---

## Ejemplo de resultado

```
▸ Modo VÍDEO (imgs/video.mp4)
▸ Detección activa. [G]=guardar  [Q/ESC]=salir

✓ Guardado: captura_20260317_174200.png  +  estado_20260317_174200.json
  🟢 Libres: 9   🔴 Ocupadas: 13
  Zona  0  Coches dentro: 5
  Zona  1  Coches dentro: 4
  Zona  2  Coches dentro: 4
```

---

## Créditos

Desarrollado para el **IES Gran Capitán** como proyecto de visión artificial aplicada a gestión de aparcamientos.

Tecnologías: [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) · OpenCV · NumPy · Python
