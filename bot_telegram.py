"""
bot_telegram.py
────────────────────────────────────────────────
Bot de Telegram para consultar el estado del aparcamiento.
Cuando el usuario pide el estado, ejecuta YOLOv8 sobre imgs/1.png
y devuelve cuántas plazas están libres.

Uso:
    python bot_telegram.py

Requisitos:
    pip install python-telegram-bot python-dotenv ultralytics

Configuración:
    Crea un fichero .env con:
        TELEGRAM_TOKEN=tu_token_aqui
────────────────────────────────────────────────
"""

import io
import os
import cv2
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from detector_yolo import (
    cargar_spots, detectar, calcular_resultados, dibujar,
    guardar_estado_actual, NUM_PLAZAS, SPOTS_JSON,
)

load_dotenv()

IMAGEN_PATH = Path("imgs/1.png")

# ── Carga única al arrancar ───────────────────
print("▸ Cargando modelo YOLO...")
from ultralytics import YOLO
model = YOLO("yolo11x.pt")

print(f"▸ Cargando plazas desde '{SPOTS_JSON}'...")
spots = cargar_spots(SPOTS_JSON)
print(f"  → {len(spots)} zonas cargadas\n")
# ─────────────────────────────────────────────


def analizar_imagen() -> tuple[dict, bytes]:
    frame = cv2.imread(str(IMAGEN_PATH))
    if frame is None:
        raise FileNotFoundError(f"No se encontró la imagen: {IMAGEN_PATH}")
    boxes      = detectar(model, frame, conf=0.30)
    resultados = calcular_resultados(spots, boxes)
    guardar_estado_actual(resultados)

    frame_viz = dibujar(frame.copy(), spots, resultados, boxes)
    _, buf = cv2.imencode(".jpg", frame_viz)
    foto = io.BytesIO(buf.tobytes())
    foto.name = "estado.jpg"

    total_coches = sum(r["coches_dentro"] for r in resultados)
    libres       = max(0, NUM_PLAZAS - total_coches)
    return {"libres": libres, "ocupadas": NUM_PLAZAS - libres, "total": NUM_PLAZAS}, foto


def formatear_respuesta(datos: dict) -> str:
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


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hola. Soy el bot del aparcamiento.\n\n"
        "Escríbeme cualquier mensaje o usa /estado para saber cuántas plazas hay libres."
    )


async def cmd_estado(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔍 Analizando aparcamiento...")
    try:
        datos, foto = analizar_imagen()
        await msg.delete()
        await update.message.reply_photo(photo=foto, caption=formatear_respuesta(datos))
    except FileNotFoundError:
        await msg.edit_text(f"⚠️ No se encontró la imagen en {IMAGEN_PATH}.")
    except Exception as e:
        await msg.edit_text(f"⚠️ Error al analizar: {e}")


async def msg_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_estado(update, context)


def main():
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("No se encontró TELEGRAM_TOKEN en .env")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("estado", cmd_estado))
    app.add_handler(CommandHandler("plazas", cmd_estado))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, msg_handler))

    print("▸ Bot activo. Pulsa Ctrl+C para parar.")
    app.run_polling()


if __name__ == "__main__":
    main()