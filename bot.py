import os
import io
import asyncio
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from aiogram import Bot, Dispatcher, F, types
from aiogram.client.default_bot_properties import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart

# ──────────────────────────────────────────────────────────────────────────────
# Конфиг
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN".lower())
if not BOT_TOKEN:
    raise RuntimeError("Env var TELEGRAM_BOT_TOKEN is empty")

# если хочешь, можешь зафиксировать здесь username канала (без @), но это не обязательно:
CHANNEL_USERNAME = "assistantdesign"  # не используется жёстко — реагируем на любые канал-посты

# Приветствие (как «раньше»)
START_TEXT = (
    "Привет! Я —  генератор цветов от ДИЗ БАЛАНС 🎨 "
    "Отправь мне фото, а я тебе отправлю его цветовую палитру в ответ."
)

# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции

def pil_get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Пытаемся взять системный шрифт, иначе — встроенный PIL."""
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def extract_dominant_colors(image: Image.Image, k: int = 12) -> List[Tuple[int, int, int]]:
    """Возвращает k доминирующих цветов (RGB)."""
    img = image.convert("RGB")
    # уменьшаем, чтобы ускорить кластеризацию
    max_side = 400
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.asarray(img).reshape(-1, 3).astype(np.float32)

    # KMeans
    kmeans = KMeans(n_clusters=k, n_init=4, random_state=42)
    labels = kmeans.fit_predict(arr)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # сортируем кластеры по размеру (чтобы первые были «доминирующими»)
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    colors = [tuple(int(x) for x in centers[i]) for i in order]
    return colors

def draw_palette(colors: List[Tuple[int, int, int]], cols: int = 3, rows: int = 4) -> Image.Image:
    """Рисуем сетку 3x4 (12 цветов) с подписями HEX."""
    assert cols * rows == len(colors), "colors length must match grid size"

    sw = 280          # ширина свотча
    sh = 220          # высота свотча без подписи
    pad = 24          # внутренние отступы
    gap = 24          # расстояние между карточками
    caption_h = 56    # высота под область подписи

    W = pad*2 + cols*sw + (cols-1)*gap
    H = pad*2 + rows*(sh+caption_h) + (rows-1)*gap

    img = Image.new("RGB", (W, H), color=(245, 245, 245))
    drw = ImageDraw.Draw(img)
    font = pil_get_font(28)

    for idx, rgb in enumerate(colors):
        r, c = divmod(idx, cols)
        x0 = pad + c*(sw + gap)
        y0 = pad + r*(sh + caption_h + gap)

        # свотч
        drw.rounded_rectangle([x0, y0, x0+sw, y0+sh], radius=16, fill=rgb)

        # подпись HEX
        hex_text = to_hex(rgb)
        # Pillow 10+: используем textbbox вместо textsize
        bbox = drw.textbbox((0, 0), hex_text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x0 + (sw - tw) // 2
        ty = y0 + sh + (caption_h - th) // 2

        # легкая «плашка» под текстом, чтобы читалось на любых цветах
        drw.rounded_rectangle(
            [x0+12, y0+sh+8, x0+sw-12, y0+sh+caption_h-8],
            radius=12, fill=(255, 255, 255)
        )
        drw.text((tx, ty), hex_text, fill=(40, 40, 40), font=font)

    return img

async def fetch_input_image(bot: Bot, message: types.Message) -> Image.Image | None:
    """Скачиваем наилучшее фото из сообщения и открываем как PIL.Image."""
    try:
        # берем самую большую вариацию
        photo_size = max(message.photo, key=lambda p: p.file_size or 0)
        file = await bot.get_file(photo_size.file_id)
        buf = io.BytesIO()
        await bot.download(file, destination=buf)
        buf.seek(0)
        return Image.open(buf)
    except Exception:
        return None

async def build_palette_image(bot: Bot, message: types.Message) -> tuple[io.BytesIO, List[str]] | None:
    """Строим палитру и возвращаем (байты PNG, список HEX)."""
    pil_img = await fetch_input_image(bot, message)
    if pil_img is None:
        return None
    try:
        colors = extract_dominant_colors(pil_img, k=12)
        palette = draw_palette(colors, cols=3, rows=4)
        out = io.BytesIO()
        palette.save(out, format="PNG", optimize=True)
        out.seek(0)
        hex_list = [to_hex(c) for c in colors]
        return out, hex_list
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Хендлеры

async def on_start(message: types.Message):
    await message.answer(START_TEXT)

async def on_photo_private(message: types.Message, bot: Bot):
    res = await build_palette_image(bot, message)
    if not res:
        await message.reply("Не удалось обработать изображение. Попробуйте другое фото.")
        return
    img_bytes, hex_list = res
    caption = "Палитра: " + " ".join(hex_list)
    await message.reply_photo(types.BufferedInputFile(img_bytes.read(), filename="palette.png"),
                              caption=caption)

async def on_photo_channel(channel_post: types.Message, bot: Bot):
    # Ответ в тот же канал, реплаем оригинальный пост
    res = await build_palette_image(bot, channel_post)
    if not res:
        await bot.send_message(
            chat_id=channel_post.chat.id,
            text="Не удалось обработать изображение. Попробуйте другое фото.",
            reply_to_message_id=channel_post.message_id
        )
        return
    img_bytes, hex_list = res
    caption = "Палитра: " + " ".join(hex_list)
    await bot.send_photo(
        chat_id=channel_post.chat.id,
        photo=types.BufferedInputFile(img_bytes.read(), filename="palette.png"),
        caption=caption,
        reply_to_message_id=channel_post.message_id
    )

# ──────────────────────────────────────────────────────────────────────────────
# Запуск

async def main():
    # НЕ передаём parse_mode старым способом
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)  # можно убрать HTML, если не нужен
    )
    dp = Dispatcher()

    # личные чаты
    dp.message.register(on_start, CommandStart())
    dp.message.register(on_photo_private, F.photo)

    # канал: бот должен быть админом канала с правом "управление сообщениями"
    dp.channel_post.register(on_photo_channel, F.photo)

    print("color-bot | Бот запущен. Ждём апдейты…")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
