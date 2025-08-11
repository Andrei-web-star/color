
import os
import io
import asyncio
import logging
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.client.default_bot_properties import DefaultBotProperties

# ── Конфиг ────────────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Нет TELEGRAM_BOT_TOKEN в переменных окружения.")

# Эти два можно оставить пустыми — они нужны только если хочешь
# жёстко дублировать ответы еще и в конкретный канал.
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "-1002608781747"))      # твой канал
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "assistantdesign")

# Сколько цветов выводим
PALETTE_SIZE = 12

# ── Инициализация ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s"
)
log = logging.getLogger("color-bot")

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()


# ── Утилиты ──────────────────────────────────────────────────────────────────
def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """Безопасно получаем размеры текста (совместимо с разными версиями Pillow)."""
    try:
        # pillow ≥ 8.0
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        try:
            # старые версии
            return draw.textsize(text, font=font)  # type: ignore
        except Exception:
            # fallback
            return font.getlength(text), font.size  # type: ignore


def dominant_colors_pil(img: Image.Image, k: int) -> List[Tuple[Tuple[int, int, int], int]]:
    """
    Выбираем k доминирующих цветов без sklearn:
    1) уменьшаем картинку
    2) используем медианный квантайзер PIL (MEDIANCUT)
    3) собираем частоты цветов и возвращаем top-k
    """
    img_small = img.convert("RGB").resize((256, 256), Image.LANCZOS)
    q = img_small.quantize(colors=k, method=Image.MEDIANCUT)
    q_rgb = q.convert("RGB")
    colors = q_rgb.getcolors(256 * 256) or []
    # сортируем по частоте убыв.
    colors.sort(key=lambda c: c[0], reverse=True)
    # colors: List[(count, (r,g,b))]
    return [(rgb, count) for count, rgb in colors[:k]]


def build_palette_image(colors: List[Tuple[Tuple[int, int, int], int]]) -> bytes:
    """
    Собираем карточку 3×4 (12 цветов) с HEX подписями.
    На выход — PNG в bytes.
    """
    # Канва
    cols = 3
    rows = 4
    sw = 520   # ширина свотча
    sh = 240   # высота свотча
    pad = 32   # внутренние отступы
    label_h = 72

    W = cols * sw + (cols + 1) * pad
    H = rows * (sh + label_h) + (rows + 1) * pad

    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    # Шрифт: используем встроенный по умолчанию
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 42)
    except Exception:
        font = ImageFont.load_default()

    for i, (rgb, _cnt) in enumerate(colors):
        r, g, b = rgb
        hex_code = rgb_to_hex((r, g, b))
        row = i // cols
        col = i % cols

        x0 = pad + col * (sw + pad)
        y0 = pad + row * (sh + label_h + pad)
        # прямоугольник цвета
        draw.rectangle([x0, y0, x0 + sw, y0 + sh], fill=(r, g, b))

        # подложка под подпись
        draw.rectangle([x0, y0 + sh, x0 + sw, y0 + sh + label_h], fill=(255, 255, 255))

        # подпись по центру
        tw, th = text_size(draw, hex_code, font)
        tx = x0 + (sw - tw) // 2
        ty = y0 + sh + (label_h - th) // 2
        # Цвет текста контрастный к фону свотча — возьмём тёмно-серый
        draw.text((tx, ty), hex_code, fill=(30, 30, 30), font=font)

    bio = io.BytesIO()
    canvas.save(bio, format="PNG")
    bio.seek(0)
    return bio.getvalue()


async def download_photo_as_image(message: types.Message) -> Image.Image:
    """
    Скачиваем наибольшее превью фото в память и открываем как PIL.Image
    """
    largest = message.photo[-1]
    buf = io.BytesIO()
    await bot.download(largest, destination=buf)
    buf.seek(0)
    return Image.open(buf)


# ── Хэндлеры ─────────────────────────────────────────────────────────────────
@dp.message(F.text == "/start")
async def cmd_start(message: types.Message):
    text = (
        "Привет! Я — генератор цветов от ДИЗ БАЛАНС 🎨 "
        "Отправь мне фото, а я тебе отправлю его цветовую палитру в ответ."
    )
    await message.answer(text)


@dp.message(F.photo)
async def on_photo(message: types.Message):
    chat = message.chat
    is_channel = chat.type == "channel"

    try:
        # 1) грузим фото
        img = await download_photo_as_image(message)

        # 2) считаем 12 доминирующих цветов
        colors = dominant_colors_pil(img, PALETTE_SIZE)
        # на всякий пожарный — добьём до 12, если вдруг меньше
        if len(colors) < PALETTE_SIZE:
            colors = (colors + colors)[:PALETTE_SIZE]

        # 3) строим карточку
        png_bytes = build_palette_image(colors)

        # 4) текст с HEX (строкой)
        hex_list = " ".join(rgb_to_hex(rgb) for rgb, _ in colors)
        caption = f"Палитра: {hex_list}"

        # 5) отправляем ответ
        photo = types.BufferedInputFile(png_bytes, filename="palette.png")

        if is_channel:
            # ответ в тред к публикации канала
            await bot.send_photo(
                chat_id=chat.id,
                photo=photo,
                caption=caption,
                reply_to_message_id=message.message_id
            )
        else:
            # личка
            await message.answer_photo(photo=photo, caption=caption)

        # опционально дублируем в конкретный канал:
        # (закомментируй, если не нужно)
        try:
            await bot.send_photo(
                chat_id=CHANNEL_ID,
                photo=types.BufferedInputFile(png_bytes, filename="palette.png"),
                caption=caption
            )
        except Exception as e:
            log.warning("Не удалось продублировать в канал: %s", e)

    except Exception as e:
        log.exception("Ошибка обработки фото: %s", e)
        err_text = "Не удалось обработать изображение. Попробуйте другое фото."
        if is_channel:
            await bot.send_message(chat.id, err_text, reply_to_message_id=message.message_id)
        else:
            await message.answer(err_text)


# ── Точка входа ──────────────────────────────────────────────────────────────
async def main():
    log.info("color-bot | Бот запускаем. Канал: @%s (id=%s)", CHANNEL_USERNAME, CHANNEL_ID)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
