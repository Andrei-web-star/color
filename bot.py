# bot.py
import asyncio
import io
import logging
import os
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, BufferedInputFile, FSInputFile

# ---------- базовая настройка ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("color-bot")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Env var TELEGRAM_BOT_TOKEN is not set")

# aiogram 3.7: parse_mode передаём через DefaultBotProperties
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# Для текста на палитре: попробуем встроенный шрифт
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 28)
except Exception:
    FONT = ImageFont.load_default()


# ---------- утилиты цвета ----------
def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def get_text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    # Pillow ≥10: используем textbbox
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def extract_palette(img: Image.Image, k: int = 4) -> List[Tuple[int, int, int]]:
    # уменьшаем для скорости/стабильности
    img_small = img.copy()
    img_small.thumbnail((300, 300))
    arr = np.array(img_small).reshape(-1, 3)

    # уберём полностью белые и полностью чёрные точки (меньше шума)
    mask = ~(
        ((arr <= 5).all(axis=1)) |
        ((arr >= 250).all(axis=1))
    )
    arr = arr[mask] if mask.any() else arr

    # кластеризация
    km = KMeans(n_clusters=k, n_init=8, random_state=42)
    km.fit(arr)
    centers = np.rint(km.cluster_centers_).astype(int)
    # сортируем по "важности" (частоте)
    counts = np.bincount(km.labels_)
    order = np.argsort(-counts)[:k]
    palette = [tuple(centers[i]) for i in order]
    return palette


def build_palette_image(palette: List[Tuple[int, int, int]], swatch_w=240, swatch_h=120, gap=4) -> Image.Image:
    k = len(palette)
    cols = min(k, 2)
    rows = (k + cols - 1) // cols

    cell_w = swatch_w
    cell_h = swatch_h
    pad = 12

    out_w = cols * cell_w + (cols + 1) * gap + pad * 2
    out_h = rows * (cell_h + 40) + (rows + 1) * gap + pad * 2  # + место под текст

    img = Image.new("RGB", (out_w, out_h), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    for idx, rgb in enumerate(palette):
        r, c = divmod(idx, cols)
        x = pad + gap + c * (cell_w + gap)
        y = pad + gap + r * (cell_h + 40 + gap)

        # прямоугольник цвета
        draw.rectangle([x, y, x + cell_w, y + cell_h], fill=rgb)

        # обводка
        draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(0, 0, 0), width=1)

        # подпись HEX
        hex_text = rgb_to_hex(rgb)
        tw, th = get_text_size(draw, hex_text, FONT)
        tx = x + (cell_w - tw) // 2
        ty = y + cell_h + 8
        # фон подписи для контраста
        draw.rounded_rectangle([tx - 6, ty - 4, tx + tw + 6, ty + th + 4], radius=6, fill=(255, 255, 255))
        draw.text((tx, ty), hex_text, font=FONT, fill=(0, 0, 0))

    return img


async def process_and_reply(message: Message) -> None:
    """
    Скачивает фото из сообщения, строит палитру и присылает её ответом.
    Работает и в ЛС, и в канале (ответом к посту).
    """
    try:
        # берём самое большое превью
        photo = message.photo[-1]
        buf = io.BytesIO()
        await bot.download(photo, destination=buf)
        buf.seek(0)

        img = Image.open(buf).convert("RGB")
        palette = extract_palette(img, k=4)
        pal_img = build_palette_image(palette)

        out = io.BytesIO()
        pal_img.save(out, format="PNG")
        out.seek(0)

        caption = "Палитра: " + "  ".join(rgb_to_hex(c) for c in palette)
        await message.reply_photo(
            BufferedInputFile(out.read(), filename="palette.png"),
            caption=caption
        )
    except Exception as e:
        log.exception("Ошибка при обработке фото")
        await message.reply("Не удалось обработать изображение. Попробуйте другое фото.")


# ---------- хендлеры ----------
@dp.message(CommandStart())
async def on_start(message: Message):
    text = (
        "Привет! Я анализирую изображения и вытаскиваю доминирующие цвета.\n\n"
        "Добавьте меня <b>админом</b> в канал <b>@assistantdesign</b>, публикуйте фото — "
        "я пришлю палитру в ответ к посту.\n\n"
        "Также можно прислать фото прямо сюда."
    )
    await message.answer(text)


# ЛС: пришлют фото
@dp.message(F.photo)
async def handle_private_photo(message: Message):
    await process_and_reply(message)


# Канал: новый фото-пост
@dp.channel_post(F.photo)
async def handle_channel_photo(message: Message):
    await process_and_reply(message)


# На всякий случай — альбомы (медиа-группы): берём последнюю фотографию
@dp.channel_post(F.media_group_id, F.photo)
async def handle_channel_album(message: Message):
    await process_and_reply(message)


# ---------- запуск ----------
async def main():
    log.info("Бот запускаем. Канал: @assistantdesign")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
