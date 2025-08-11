# bot.py
import os
import io
import asyncio
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sklearn.cluster import KMeans

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Message
from aiogram.filters import CommandStart

# ====== НАСТРОЙКИ КАНАЛА (можно светить в GitHub) ======
CHANNEL_USERNAME = "assistantdesign"     # без @
CHANNEL_ID = -1002608787147              # id канала

# ====== ТОКЕН (НЕ СВЕТИМ) ======
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Не найдена переменная окружения TELEGRAM_BOT_TOKEN")

# aiogram 3.7+: parse_mode передаём через DefaultBotProperties
bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()
r = Router()
dp.include_router(r)

# ====== ТЕКСТ ПРИВЕТСТВИЯ ======
HELLO = (
    "Привет! Я —  генератор цветов от ДИЗ БАЛАНС 🎨 "
    "Отправь мне фото, а я тебе отправлю его цветовую палитру в ответ."
)

# ====== УТИЛИТЫ ЦВЕТОВ ======
def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def text_color_for(bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
    # YIQ контраст для белого/чёрного текста
    r, g, b = bg
    yiq = (r*299 + g*587 + b*114) / 1000
    return (0, 0, 0) if yiq > 160 else (255, 255, 255)

def kmeans_palette(img: Image.Image, k: int = 12) -> List[Tuple[int, int, int]]:
    # уменьшаем картинку для ускорения
    img_small = img.copy().convert("RGB")
    img_small.thumbnail((400, 400), Image.LANCZOS)
    pixels = np.array(img_small).reshape(-1, 3)

    # KMeans
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_.astype(int)

    # частоты кластеров — чтобы отсортировать доминантные
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(counts)[::-1]  # по убыванию частоты

    palette = [tuple(int(x) for x in centers[i]) for i in order]
    return palette

def draw_palette_image(palette: List[Tuple[int, int, int]]) -> io.BytesIO:
    """
    Рисуем карточку 12 цветов 3x4 с хекс‑подписями.
    Возвращаем буфер готовой PNG‑картинки.
    """
    # Сетка 3 колонки x 4 строки
    cols, rows = 3, 4
    cell_w, cell_h = 360, 220
    pad = 24
    width = cols * cell_w + (cols + 1) * pad
    height = rows * cell_h + (rows + 1) * pad + 16  # немного запаса снизу

    img = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Шрифт: попробуем системный, иначе — дефолтный PIL
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 34)
    except Exception:
        font = ImageFont.load_default()

    for idx, color in enumerate(palette[:12]):
        r_i = idx // cols
        c_i = idx % cols

        x0 = pad + c_i * (cell_w + pad)
        y0 = pad + r_i * (cell_h + pad)
        x1 = x0 + cell_w
        y1 = y0 + cell_h

        # сам цветной блок
        draw.rectangle([x0, y0, x1, y1 - 60], fill=color)

        # подпись HEX снизу блока
        hex_code = rgb_to_hex(color)
        text_w, text_h = draw.textsize(hex_code, font=font)
        tx = x0 + (cell_w - text_w) // 2
        ty = y1 - 52

        # фон ярлычка (слегка серый)
        tag_pad = 10
        draw.rounded_rectangle(
            [tx - tag_pad, ty - tag_pad, tx + text_w + tag_pad, ty + text_h + tag_pad],
            radius=8,
            fill=(255, 255, 255)
        )
        draw.text((tx, ty), hex_code, fill=(30, 30, 30), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf

async def analyze_and_reply(msg: Message, image: Image.Image) -> None:
    try:
        palette = kmeans_palette(image, k=12)
        card = draw_palette_image(palette)
        hex_list = " ".join(rgb_to_hex(c) for c in palette[:12])
        await bot.send_photo(
            chat_id=msg.chat.id,
            photo=card,
            caption=f"Палитра: {hex_list}",
            reply_to_message_id=msg.message_id
        )
    except Exception:
        await msg.reply("Не удалось обработать изображение. Попробуйте другое фото.")

# ====== ОБРАБОТЧИКИ ======
@r.message(CommandStart())
async def on_start(msg: Message):
    await msg.answer(HELLO)

@r.message(F.photo)
async def on_private_photo(msg: Message):
    # берём самое большое фото из массива
    file_id = msg.photo[-1].file_id
    file = await bot.get_file(file_id)
    buf = io.BytesIO()
    await bot.download(file, buf)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    await analyze_and_reply(msg, img)

@r.channel_post(F.photo)
async def on_channel_photo(msg: Message):
    """
    Бот должен быть админом канала.
    Отвечаем прямо под постом в канале.
    """
    # убеждаемся, что это нужный канал (если бот в нескольких)
    if msg.chat.id not in (CHANNEL_ID,):
        return

    file_id = msg.photo[-1].file_id
    file = await bot.get_file(file_id)
    buf = io.BytesIO()
    await bot.download(file, buf)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    # генерим и отвечаем в канал
    try:
        palette = kmeans_palette(img, k=12)
        card = draw_palette_image(palette)
        hex_list = " ".join(rgb_to_hex(c) for c in palette[:12])
        await bot.send_photo(
            chat_id=CHANNEL_ID,
            photo=card,
            caption=f"Палитра: {hex_list}",
            reply_to_message_id=msg.message_id
        )
    except Exception:
        await bot.send_message(
            chat_id=CHANNEL_ID,
            text="Не удалось обработать изображение. Попробуйте другое фото.",
            reply_to_message_id=msg.message_id
        )

async def main():
    # Просто лог‑сообщение в логи Render’а
    print(f"color-bot | Бот запускаем. Канал: @{CHANNEL_USERNAME}")
    await dp.start_polling(bot, allowed_updates=["message", "channel_post"])

if __name__ == "__main__":
    asyncio.run(main())
