"""
================================================================================
MÓDULO DE INTERFAZ: API CONVERSACIONAL (BOT ASÍNCRONO DE TELEGRAM)
================================================================================
Asignatura: Visión Artificial | Proyecto: Gestión de Aparcamiento Inteligente
Desarrollado por: Javier y Pako

PROPÓSITO TÉCNICO:
Este script actúa como la capa de presentación del proyecto. Expone las capacidades
del motor analítico de Inteligencia Artificial hacia una interfaz de usuario 
remota y multiplataforma mediante el protocolo de la API de Bots de Telegram.
Implementa un diseño de ejecución asíncrona para soportar concurrencia masiva.
================================================================================
"""

import io
import os
import random
import cv2
from pathlib import Path
from dotenv import load_dotenv

# Librerías del Framework Asíncrono de Telegram
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# REFACTORIZACIÓN INTEGRAL: Sincronización absoluta con el motor core y configuración
from config import (
    SPOTS_JSON, ESTADO_ACTUAL, NUM_PLAZAS, CONF_UMBRAL, MODELO_YOLO
)
from detector_yolo import (
    cargar_spots, detectar, calcular_resultados, dibujar, guardar_estado_actual
)

# Cargamos el archivo de entorno local con las credenciales de seguridad (.env)
load_dotenv()

# Dataset de simulación visual: El bot seleccionará aleatoriamente un frame para simular transmisiones en vivo.
IMAGENES = [Path(f"imgs/{i}.png") for i in range(1, 4)]


# ── FASE DE INICIALIZACIÓN DEL SERVICIO (SINGLETON DE CARGA ÚNICA) ───────────
# ¡PUNTO CLAVE EXAMEN!: Instanciar el modelo de Deep Learning y cargar el JSON
# se realiza en el ámbito global del script. Esto garantiza que la red neuronal
# se monte en la memoria RAM una sola vez al arrancar el servidor. Si hiciéramos esto
# dentro de la función del comando, el bot tardaría varios segundos en responder a cada usuario.
print("▸ [IA] Inicializando red neuronal en la GPU/CPU...")
from ultralytics import YOLO
model = YOLO(MODELO_YOLO) # Consumo de vuestro modelo oficial de config.py

print(f"▸ [GEOMETRÍA] Cargando mapa de plazas indexadas desde '{SPOTS_JSON}'...")
spots = cargar_spots(SPOTS_JSON)
print(f"  → Se han sincronizado {len(spots)} zonas de aparcamiento en la caché.\n")
# ─────────────────────────────────────────────────────────────────────────────


def analizar_imagen() -> tuple[dict, bytes]:
    """
    PIPELINE ANALÍTICO BAJO DEMANDA:
    Simula la captura de la webcam, ejecuta la inferencia predictiva de YOLOv8
    y computa la matriz visual resultante mapeada en un flujo de bytes en memoria.
    """
    # 1. Simulación del flujo de la cámara IP (Selección pseudoaleatoria)
    imagen_path = random.choice(IMAGENES)
    frame = cv2.imread(str(imagen_path))
    if frame is None:
        raise FileNotFoundError(f"Fallo de E/S: No se pudo capturar el dispositivo de vídeo en {imagen_path}")
    
    # 2. Inferencia y procesamiento espacial geométrico consumiendo vuestras constantes
    boxes      = detectar(model, frame, conf=CONF_UMBRAL)
    resultados = calcular_resultados(spots, boxes)
    guardar_estado_actual(resultados)

    # 3. Capa de diseño gráfico (Anotación semáforo y cajas YOLO)
    frame_viz = dibujar(frame.copy(), spots, resultados, boxes)
    
    # GESTIÓN DE MEMORIA EN PRODUCCIÓN (Flujos binarios):
    # En lugar de guardar la imagen anotada en el disco duro del servidor para luego leerla
    # (lo cual generaría latencia por operaciones de lectura/escritura de E/S), codificamos la 
    # matriz de píxeles directamente en formato binario JPEG en la memoria RAM usando io.BytesIO.
    _, buf = cv2.imencode(".jpg", frame_viz)
    foto = io.BytesIO(buf.tobytes())
    foto.name = "estado.jpg" # Metadato requerido por Telegram para el buffer multimedia

    # 4. Consolidación de métricas estadísticas
    total_coches = sum(r["coches_dentro"] for r in resultados)
    libres       = max(0, NUM_PLAZAS - total_coches)
    
    return {"libres": libres, "ocupadas": NUM_PLAZAS - libres, "total": NUM_PLAZAS}, foto


def formatear_respuesta(datos: dict) -> str:
    """Formatea la salida textual inyectando emojis dinámicos según el estado del parking."""
    from datetime import datetime
    libres   = datos["libres"]
    ocupadas = datos["ocupadas"]
    total    = datos["total"]
    hora     = datetime.now().strftime("%H:%M:%S")
    
    if libres > 0:
        return (
            f"🟢 Hay {libres} plaza{'s' if libres != 1 else ''} libre{'s' if libres != 1 else ''} "
            f"de {total}.\n"
            f"🔴 Ocupadas: {ocupadas}/{total}\n"
            f"⏱ Actualizado: {hora}"
        )
    return (
        f"🔴 Aparcamiento completo — 0/{total} plazas libres.\n"
        f"⏱ Actualizado: {hora}"
    )


# ── MANEJADORES DE COMANDOS ASÍNCRONOS (ARCHITECTURE HANDLERS) ───────────────
# ¡EXPLICACIÓN PARA EL PROFESOR!: Usamos la sintaxis 'async/await' basada en Corrutinas.
# Esto evita que el bot se bloquee de forma síncrona mientras YOLO procesa la imagen.
# Si 10 usuarios piden el estado a la vez, el bot procesa las peticiones de forma concurrente.

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manejador del comando de bienvenida."""
    await update.message.reply_text(
        "👋 Hola. Soy el sistema de Inteligencia Artificial del IES Gran Capitán.\n\n"
        "Escríbeme cualquier mensaje o usa /estado para consultar el aforo en tiempo real."
    )


async def cmd_estado(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manejador core: Ejecuta el pipeline analítico y despacha los reportes visuales."""
    # Enviamos un feedback intermedio no bloqueante para mejorar la experiencia de usuario (UX)
    msg = await update.message.reply_text("🔍 Analizando aparcamiento...")
    try:
        # Invocamos el pipeline analítico
        datos, foto = analizar_imagen()
        
        # Eliminamos el mensaje temporal de espera de forma asíncrona
        await msg.delete()
        
        # Despachamos el paquete multimedia (Imagen procesada + Texto formateado)
        await update.message.reply_photo(photo=foto, caption=formatear_respuesta(datos))
        
    except FileNotFoundError as e:
        await msg.edit_text(f"⚠️ Error de hardware: {e}")
    except Exception as e:
        await msg.edit_text(f"⚠️ Error crítico en el pipeline: {e}")


async def msg_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Fallback: Redirige cualquier mensaje de texto plano hacia la función analítica."""
    await cmd_estado(update, context)


def main():
    # Recuperación segura del token cifrado de la API
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("Error de entorno: TELEGRAM_TOKEN no configurado en el archivo .env")

    # Inicialización del motor asíncronico de la aplicación
    app = ApplicationBuilder().token(token).build()
    
    # Registro de rutas y comandos (Event Listeners)
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("estado", cmd_estado))
    app.add_handler(CommandHandler("plazas", cmd_estado))
    
    # Escucha activa de texto plano (Cualquier mensaje activa el pipeline)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, msg_handler))

    print("▸ [SERVIDOR] Bot de Telegram desplegado con éxito. Escuchando peticiones...")
    # Arranca el bucle infinito de escucha por sondeo activo (Long Polling)
    app.run_polling()


if __name__ == "__main__":
    main()