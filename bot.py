# bot.py
import asyncio
import logging
import os
from io import BytesIO
from typing import List, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, Update
from aiogram.client.session.middlewares.request_logging import logger as aio_logger
from aiogram.exceptions import TelegramConflictError, TelegramUnauthorizedError

# ------------------ Конфиг ------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN".upper()) or os.getenv("TOKEN")
CHANNEL_USERNAME = "@assistantdesign"          # публичный @username канала
CHANNEL_ID = -1002608781747                    # numeric id канала (оставьте как у вас)

# Безопасный parse_mode для aiogram 3.7+: задаётся в DefaultBotProperties на старте
DEFAULT_PARSE_MODE = ParseMode.HTML

# ------------------ Логирование ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(name)s: %(message)s"
)
log = logging.getLogger("color-bot")
aio_logger.setLevel(logging.WARNING)

# ------------------ Утилиты ------------------
def fetch_file_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def extract_palette(img: Image.Image, k: int = 4) -> List[Tuple[int, int, int]]:
    # уменьшаем для скорости, убираем альфу
    img = img.convert("RGB").resize((256, int(256 * img.height / max(1, img.width))))
    data = np.array(img).reshape(-1, 3)

    # иногда попадаются почти-одноцветные фото — KMeans может падать
    unique = np.unique(data, axis=0)
    k = min(k, len(unique))
    if k < 2:
        return [tuple(int(v) for v in unique[0])]  # один цвет

    km = KMeans(n_clusters=k, n_init=3, random_state=42)
    km.fit(data)
    centers = km.cluster_centers_.astype(int)
    return [tuple(map(int, c)) for c in centers]

def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    """
    Универсальный способ посчитать размер текста для Pillow 9/10/11:
    сначала пробуем textbbox, если нет — textlength/textsize.
    """
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    # запасной путь
    if hasattr(draw, "textlength"):
        w = int(draw.textlength(text, font=font))
        # высоту берём из bbox шрифта
        try:
            _, _, _, b = font.getbbox("Hg")  # примерно x-height
            h = b
        except Exception:
            h = font.size + 4
        return w, h
    # совсем старый Pillow
    return draw.textsize(text, font=font)

def hex_color(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def render_palette(colors: List[Tuple[int, int, int]]) -> Image.Image:
    sw = 220              # ширина одного сэмпла
    h = 180               # высота карточки
    gap = 12              # отступ между блоками
    pad = 20
    total_w = pad * 2 + sw * len(colors) + gap * (len(colors) - 1)
    img = Image.new("RGB", (total_w, h), (245, 245, 245))
    d = ImageDraw.Draw(img)

    # шрифт
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    x = pad
    for c in colors:
        # прямоугольник цвета
        rect_h = h - 60
        d.rectangle([x, pad, x + sw, pad + rect_h], fill=c)

        # подписи
        rgb_text = f"{c[0]},{c[1]},{c[2]}"
        hex_text = hex_color(c)

        w1, _ = _text_size(d, rgb_text, font)
        w2, _ = _text_size(d, hex_text, font)

        d.text((x + (sw - w1) // 2, pad + rect_h + 8), rgb_text, fill=(0, 0, 0), font=font)
        d.text((x + (sw - w2) // 2, pad + rect_h + 34), hex_text, fill=(0, 0, 0), font=font)

        x += sw + gap

    return img

async def reply_palette(bot: Bot, target_chat_id: int, reply_to_message_id: int, img_bytes: bytes):
    with Image.open(BytesIO(img_bytes)) as im:
        colors = extract_palette(im, k=4)
    palette_img = render_palette(colors)
    bio = BytesIO()
    palette_img.save(bio, format="PNG")
    bio.seek(0)
    await bot.send_photo(
        target_chat_id,
        bio,
        caption="Палитра доминирующих цветов",
        reply_to_message_id=reply_to_message_id
    )

# ------------------ Хендлеры ------------------
async def on_start(message: Message, bot: Bot):
    text = (
        "Привет! Я анализирую изображения и вытаскиваю доминирующие цвета.\n\n"
        f"Добавьте меня админом в канал {CHANNEL_USERNAME}, публикуйте фото — "
        "я пришлю палитру в ответ к посту."
    )
    await message.answer(text)

async def on_channel_photo(message: Message, bot: Bot):
    try:
        # берём самую большую версию фото
        if not message.photo:
            return
        file_id = message.photo[-1].file_id
        f = await bot.get_file(file_id)
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{f.file_path}"
        img = fetch_file_bytes(file_url)

        await reply_palette(bot, message.chat.id, message.message_id, img)
    except Exception as e:
        log.exception("Failed to process channel photo")
        await bot.send_message(
            message.chat.id,
            "Не удалось обработать изображение. Попробуйте другое фото.",
            reply_to_message_id=message.message_id
        )

async def on_private_photo(message: Message, bot: Bot):
    # тот же сценарий, но в личке
    try:
        file_id = message.photo[-1].file_id
        f = await bot.get_file(file_id)
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{f.file_path}"
        img = fetch_file_bytes(file_url)
        await reply_palette(bot, message.chat.id, message.message_id, img)
    except Exception:
        log.exception("Failed to process direct photo")
        await message.answer("Не удалось обработать изображение. Попробуйте другое фото.")

# ------------------ Main ------------------
async def run():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не найден в Environment Variables.")

    # aiogram 3.7+: parse_mode задаётся в default properties
    bot = Bot(BOT_TOKEN, default=Bot.default_bot_properties(parse_mode=DEFAULT_PARSE_MODE))
    dp = Dispatcher()

    # команды
    dp.message.register(on_start, CommandStart())

    # канал: реагируем только на наш канал
    dp.message.register(
        on_channel_photo,
        F.chat.type == "channel",
        F.chat.username.as_("uname").map(lambda u: f"@{u}".lower() == CHANNEL_USERNAME.lower()),
        F.photo
    )

    # личка
    dp.message.register(on_private_photo, F.chat.type == "private", F.photo)

    log.info("бот запущен. Канал: %s (id=%s)", CHANNEL_USERNAME, CHANNEL_ID)

    # Защита от двойного запуска (409 Conflict) — просто повторяем старт, если инстанс уже есть
    retry = 0
    while True:
        try:
            await dp.start_polling(bot, allowed_updates=Update.ALL_TYPES)
        except TelegramConflictError:
            retry += 1
            wait = min(5, 0.8 + retry * 0.6)
            log.warning("Conflict: уже есть активный инстанс. Ждём %.1fs и пробуем ещё…", wait)
            await asyncio.sleep(wait)
        except TelegramUnauthorizedError:
            log.error("Bot token неверный или отозван.")
            break
        except Exception:
            log.exception("Unhandled error in polling. Перезапуск через 2s…")
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(run())
