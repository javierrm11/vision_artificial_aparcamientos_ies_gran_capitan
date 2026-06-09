"""
================================================================================
MÓDULO DE CONFIGURACIÓN CENTRALIZADA (SISTEMA GLOBAL)
================================================================================
Este archivo centraliza todas las constantes, rutas de persistencia y 
parámetros analíticos del sistema con los valores calibrados por el equipo.
================================================================================
"""

from pathlib import Path

# ── CONFIGURACIÓN DE ARCHIVOS Y RUTAS DE CONTROL ORIGINALES ──────────────────
SPOTS_JSON        = Path("imgs/spots.json")     # Fichero de coordenadas geométricas ROIs
OUTPUT_DIR        = Path("imgs/capturas")       # Almacén de logs visuales
ESTADO_ACTUAL     = Path("imgs/estado_actual.json") # Estado numérico para el bot
NUM_PLAZAS        = 22                          # Capacidad nominal del parking del centro

# ── PARÁMETROS ANALÍTICOS DE INTELIGENCIA ARTIFICIAL ORIGINALES ──────────────
CONF_UMBRAL       = 0.15                        # Confianza mínima YOLO adaptada al entorno
SOLAPAMIENTO_MIN  = 0.10                        # Fracción mínima del bbox dentro del ROI (0.0-1.0)
MODELO_YOLO       = "yolo11x.pt"                # Modelo base 11x

# ── PALETA CROMÁTICA GLOBAL BGR (OpenCV) ORIGINAL ────────────────────────────
COL_LIBRE         = (0, 220, 80)                # Verde
COL_OCUPADA       = (0, 60, 220)                # Rojo/Marrón original
COL_YOLO          = (0, 165, 255)               # Naranja para cajas YOLO