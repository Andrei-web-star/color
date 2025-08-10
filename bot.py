import os
import io
import asyncio
import logging
import random
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramConflictError, TelegramUnauthorizedError
from aiogram.client.default import DefaultBotProperties  # <— добавили

# -------------------
# НАСТРОЙКИ КАНАЛА
# -------------------
CHANNEL_USERNAME = "assistantdesign"      # @assistantdesign (без @)
CHANNEL_ID_HINT = -1002608787147

PALETTE_SIZE = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

def _hex(c: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*c)

def extract_palette(img: Image.Image, k: int = PALETTE_SIZE) -> List[Tuple[int, int, int]]:
    work = img.convert("RGB")
    work.thumbnail((400, 400))
    arr = np.array(work).reshape(-1, 3).astype(np.float32)
    arr += np.random.normal(0, 1.0, arr.shape)
    arr = np.clip(arr, 0, 255)

    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    km.fit(arr)
    centers = km.cluster_centers_.astype(int)
    counts = np.bincount(km.labels_)
    order = np.argsort(counts)[::-1]
    return [tuple(centers[i]) for i in order]

def draw_palette(palette: List[Tuple[int, int, int]]) -> Image.Image:
    sw, sh, gap, text_h = 220, 180, 10, 36
    width = sw * len(palette) + gap * (len(palette) + 1)
    height = sh + text_h + gap * 2
    im = Image.new("RGB", (width, height), (245, 245, 245))
    drw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    x = gap
    for rgb in palette:
        drw.rectangle([x, gap, x + sw, gap + sh], fill=rgb)
        hex_code = _hex(rgb)
        bbox = drw.textbbox((0, 0), hex_code, font=font)  # Pillow 10+
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x + (sw - tw) // 2
        ty = gap + sh + (text_h - th) // 2
        drw.rectangle([x, gap + sh, x + sw, gap + sh + text_h], fill=(255, 255, 255))
        drw.text((tx, ty), hex_code, fill=(30, 30, 30), font=font)
        x += sw + gap
    return im

async def make_palette_photo(bot: Bot, message: types.Message, file_id: str):
    buf = io.BytesIO()
    await bot.download(file_id, buf)
    buf.seek(0)
    img = Image.open(buf)
    palette = extract_palette(img, PALETTE_SIZE)
    palette_img = draw_palette(palette)
    out = io.BytesIO()
    palette_img.save(out, format="PNG")
    out.seek(0)
    caption = "Палитра из {} цветов:\n{}".format(
        PALETTE_SIZE, "\n".join(_hex(c) for c in palette)
    )
    await message.answer_photo(
        photo=types.BufferedInputFile(out.read(), filename="palette.png"),
        caption=caption
    )

async def on_start(message: types.Message):
    text = (
        "Привет! Я анализирую изображения и вытаскиваю доминирующие цвета.\n\n"
        f"Добавьте меня админом в канал @{CHANNEL_USERNAME}, публикуйте фото — "
        "я пришлю палитру в ответ к посту."
    )
    await message.answer(text)

async def on_private_photo(message: types.Message, bot: Bot):
    await make_palette_photo(bot, message, message.photo[-1].file_id)

async def on_channel_photo(channel_post: types.Message, bot: Bot):
    await make_palette_photo(bot, channel_post, channel_post.photo[-1].file_id)

def _token() -> str:
    t = os.getenv("TELEGRAM_BOT_TOKEN")
    if not t:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не найден в переменных окружения.")
    return t

async def run():
    bot = Bot(  # <— исправили создание бота
        token=_token(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()
    dp.message.register(on_start, CommandStart())
    dp.message.register(on_private_photo, F.photo)
    dp.channel_post.register(on_channel_photo, F.photo)

    try:
        await bot.delete_webhook(drop_pending_updates=True)
    except Exception as e:
        logging.warning("delete_webhook warning: %s", e)

    while True:
        try:
            logging.info("Start polling")
            await dp.start_polling(
                bot,
                allowed_updates=dp.resolve_used_update_types(),
                handle_signals=False,
            )
        except TelegramConflictError:
            wait = round(random.uniform(1.5, 3.5), 2)
            logging.warning("409 Conflict. Сплю %.2f сек и пробую снова…", wait)
            await asyncio.sleep(wait)
        except TelegramUnauthorizedError:
            logging.error("Unauthorized: проверь TELEGRAM_BOT_TOKEN.")
            await asyncio.sleep(5)
        except Exception as e:
            logging.exception("Unhandled error in polling: %s", e)
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(run())
