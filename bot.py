# bot.py
import os
import io
import asyncio
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aiogram import Bot, Dispatcher, F, Router, types
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart

# ====== НАСТРОЙКИ КАНАЛА (как просили — вписаны явно) ======
CHANNEL_USERNAME = "assistantdesign"        # t.me/assistantdesign
CHANNEL_ID = -10020628787147                # числовой id канала

# ====== ЗАГРУЗКА ТОКЕНА И СОЗДАНИЕ БОТА ======
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Env var TELEGRAM_BOT_TOKEN is missing")

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)  # aiogram 3.7+
)

dp = Dispatcher()
router = Router()
dp.include_router(router)

# ====== УТИЛИТЫ ДЛЯ ПАЛИТРЫ ======

def _to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb).lower()

def extract_palette(img: Image.Image, k: int = 5) -> List[Tuple[int, int, int]]:
    """
    Берём k доминирующих цветов через KMeans (scikit-learn),
    предварительно уменьшаем изображение для скорости.
    """
    # лёгкая нормализация
    img = img.convert("RGB")
    img_small = img.resize((300, 300))
    arr = np.array(img_small).reshape(-1, 3).astype(np.float32)

    # KMeans (склейка оттенков)
    # Склеим кластеры детерминированно для повторяемости
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(arr)
    centers = km.cluster_centers_.astype(np.uint8)

    # Отсортируем по размеру кластера (частоте вхождений)
    counts = np.bincount(labels)
    order = np.argsort(counts)[::-1]

    palette = [tuple(map(int, centers[i])) for i in order]
    return palette

def draw_palette_card(palette: List[Tuple[int, int, int]]) -> Image.Image:
    """
    Рисуем карточку 1000x560: сверху полосы цветов, снизу подписи HEX.
    """
    width, height = 1000, 560
    pad = 30
    swatch_h = 320
    gap = 12

    card = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(card)

    n = len(palette)
    swatch_w = (width - pad*2 - gap*(n-1)) // n

    # Полосы цветов
    x = pad
    for rgb in palette:
        draw.rectangle([x, pad, x + swatch_w, pad + swatch_h], fill=rgb)
        x += swatch_w + gap

    # Подписи HEX под каждой полосой
    # Подберём простой системный шрифт (на сервере не гарантированы TTF)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 32)
    except:
        font = ImageFont.load_default()

    x = pad
    y_text = pad + swatch_h + 40

    for rgb in palette:
        hex_code = _to_hex(rgb)
        # рамка под текст, чтобы читалось
        text_w, text_h = draw.textbbox((0, 0), hex_code, font=font)[2:]
        cx = x + swatch_w // 2
        tx = cx - text_w // 2
        draw.text((tx, y_text), hex_code, fill=(30, 30, 30), font=font)
        x += swatch_w + gap

    return card

# ====== ХЕНДЛЕРЫ ======

@router.message(CommandStart())
async def cmd_start(msg: types.Message):
    text = (
        "Привет! Я анализирую изображения и вытаскиваю доминирующие цвета.\n\n"
        f"Добавьте меня админом в канал <b>@{CHANNEL_USERNAME}</b>, "
        "публикуйте фото — я пришлю палитру в ответ к посту."
    )
    await msg.answer(text)

@router.message(F.chat.type == "channel", F.photo)
async def on_channel_photo(msg: types.Message):
    # Обрабатываем ТОЛЬКО наш канал
    if msg.chat.id != CHANNEL_ID:
        return

    try:
        # Берём самое большое превью
        file_id = msg.photo[-1].file_id

        # Скачиваем в память
        buf = io.BytesIO()
        await bot.download(file_id, destination=buf)
        buf.seek(0)

        # Анализ цвета
        img = Image.open(buf)
        palette = extract_palette(img, k=5)
        card = draw_palette_card(palette)

        # Готовим буфер с картинкой‑карточкой
        out = io.BytesIO()
        card.save(out, format="PNG")
        out.seek(0)

        # Текст подписи
        hex_list = " • ".join(_to_hex(rgb) for rgb in palette)
        caption = f"Палитра: {hex_list}"

        # Отправляем в ответ на пост
        await bot.send_photo(
            chat_id=msg.chat.id,
            photo=out,
            caption=caption,
            reply_to_message_id=msg.message_id
        )

    except Exception as e:
        # Лог в канал (в ответ) — чтобы видеть причину, если что-то пойдёт не так
        await bot.send_message(
            chat_id=msg.chat.id,
            text=f"⚠️ Ошибка при обработке изображения: <code>{e}</code>",
            reply_to_message_id=msg.message_id
        )

# (Необязательно) можно слушать личку, если кто-то пришлёт фото боту напрямую
@router.message(F.chat.type.in_({"private"}), F.photo)
async def on_private_photo(msg: types.Message):
    file_id = msg.photo[-1].file_id
    buf = io.BytesIO()
    await bot.download(file_id, destination=buf)
    buf.seek(0)

    img = Image.open(buf)
    palette = extract_palette(img, k=5)
    card = draw_palette_card(palette)

    out = io.BytesIO()
    card.save(out, format="PNG")
    out.seek(0)

    hex_list = "\n".join(_to_hex(rgb) for rgb in palette)
    await msg.answer_photo(out, caption=f"Палитра:\n{hex_list}")

# ====== ЗАПУСК ПОЛЛИНГА ======

async def main():
    await dp.start_polling(
        bot,
        allowed_updates=dp.resolve_used_update_types()
    )

if __name__ == "__main__":
    asyncio.run(main())
