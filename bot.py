
import os
import io
import asyncio
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message
from aiogram.enums import ParseMode

# ===== Канал (публичный username и числовой ID) =====
CHANNEL_USERNAME = "assistantdesign"      # @assistantdesign
CHANNEL_ID = -1002260787747               # -100...

# ===== Бот =====
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Env var TELEGRAM_BOT_TOKEN is not set")

bot = Bot(BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()
router = Router()
dp.include_router(router)

# ---------- k-means на чистом NumPy (без sklearn) ----------
def kmeans_colors(pixels: np.ndarray, k: int = 5, iters: int = 12, seed: int = 42) -> np.ndarray:
    """
    pixels: (N,3) uint8
    return: (k,3) float64 центроиды
    """
    rng = np.random.default_rng(seed)

    # k-means++ (упрощённо)
    centroids = pixels[rng.choice(len(pixels), size=1, replace=False)].astype(np.float64)
    for _ in range(1, k):
        d2 = np.min(((pixels[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1)
        s = d2.sum()
        probs = d2 / s if s > 0 else np.ones_like(d2) / len(d2)
        centroids = np.vstack([centroids, pixels[rng.choice(len(pixels), p=probs)].astype(np.float64)])

    for _ in range(iters):
        d2 = ((pixels[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)

        new_centroids = []
        for j in range(k):
            cluster = pixels[labels == j]
            if len(cluster) == 0:
                new_centroids.append(pixels[rng.integers(0, len(pixels))].astype(np.float64))
            else:
                new_centroids.append(cluster.mean(axis=0))
        new_centroids = np.vstack(new_centroids)

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb).lower()

def build_palette_image(colors: List[Tuple[int,int,int]], size=(900, 320)) -> Image.Image:
    w, h = size
    n = len(colors)
    pad = 4
    text_h = 48
    sw_h = h - text_h - pad*3
    sw_w = (w - pad*(n+1)) // n

    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    # подобрать шрифт (с запасными вариантами)
    font = None
    for name in ["DejaVuSans.ttf", "Arial.ttf"]:
        try:
            font = ImageFont.truetype(name, 22)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    for i, c in enumerate(colors):
        x0 = pad + i*(sw_w + pad)
        y0 = pad
        x1 = x0 + sw_w
        y1 = y0 + sw_h
        draw.rectangle([x0, y0, x1, y1], fill=tuple(c))

        hex_code = rgb_to_hex(tuple(c))
        text = hex_code
        tw, th = draw.textbbox((0,0), text, font=font)[2:]
        draw.text((x0 + (sw_w - tw)//2, y1 + pad), text, fill="black", font=font)

    return img

async def process_photo(message: Message):
    # берём максимальное фото
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    data = await bot.download_file(file.file_path)
    img = Image.open(io.BytesIO(data.read())).convert("RGB")

    # уменьшим, чтобы ускорить кластеризацию
    max_side = 512
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))

    arr = np.array(img, dtype=np.uint8)
    pixels = arr.reshape(-1, 3)

    # кластеризация
    k = 5
    centroids = kmeans_colors(pixels, k=k)

    # пересчитаем близость для подсчёта долей (чтобы отсортировать по популярности)
    d2 = ((pixels[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(d2, axis=1)
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)  # по убыванию

    top = [tuple(np.clip(centroids[i].round().astype(int), 0, 255)) for i in order]

    # картинка-палитра
    palette_img = build_palette_image(top)
    buf = io.BytesIO()
    palette_img.save(buf, format="PNG")
    buf.seek(0)

    # подпись
    hex_list = [rgb_to_hex(c) for c in top]
    caption = "Палитра: " + "  ".join(hex_list)

    # отправляем туда, откуда пришло
    await message.reply_photo(buf, caption=caption)

# ---------- Хэндлеры ----------

@router.message(F.text == "/start")
async def on_start(message: Message):
    await message.answer(
        "Привет! Пришлите фото — я сделаю палитру (HEX).\n"
        "Работаю и в личке, и в канале @assistantdesign (нужно быть подписчиком)."
    )

# Личка: фото
@router.message(F.chat.type == "private", F.photo)
async def on_private_photo(message: Message):
    await process_photo(message)

# Канал: фото (бот должен быть админом с правом «Управление сообщениями»)
@router.message(F.chat.type == "channel", F.photo)
async def on_channel_photo(message: Message):
    # дополнительная страховка: обрабатываем только наш канал
    if message.chat.id in (CHANNEL_ID,):
        await process_photo(message)

async def main():
    print("Bot is starting...")
    await dp.start_polling(bot, allowed_updates=["message"])

if __name__ == "__main__":
    asyncio.run(main())
