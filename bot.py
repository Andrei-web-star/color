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
from aiogram.filters import CommandStart
from aiogram.exceptions import TelegramConflictError, TelegramUnauthorizedError

# -------------------
# НАСТРОЙКИ КАНАЛА
# -------------------
CHANNEL_USERNAME = "assistantdesign"      # @assistantdesign (без @)
# Бот будет работать и по username, и по id. ID не обязателен, но оставляю для логов.
CHANNEL_ID_HINT = -1002608787147          # если поменяется — не страшно, код не зависит

# Сколько доминирующих цветов рисуем
PALETTE_SIZE = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

def _hex(c: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*c)

def extract_palette(img: Image.Image, k: int = PALETTE_SIZE) -> List[Tuple[int, int, int]]:
    """KMeans по уменьшенной копии изображения."""
    # делаем копию, уменьшаем для скорости и стабильности
    work = img.convert("RGB")
    work.thumbnail((400, 400))
    arr = np.array(work).reshape(-1, 3)

    # иногда совсем тёмные/светлые пиксели «шумят» — слегка подмешаем jitter
    arr = arr.astype(np.float32)
    arr += np.random.normal(0, 1.0, arr.shape)
    arr = np.clip(arr, 0, 255)

    # KMeans
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    km.fit(arr)
    centers = km.cluster_centers_.astype(int)

    # сортируем по встречаемости (по размеру кластера)
    counts = np.bincount(km.labels_)
    order = np.argsort(counts)[::-1]
    palette = [tuple(centers[i]) for i in order]
    return palette

def draw_palette(palette: List[Tuple[int, int, int]]) -> Image.Image:
    """Рисует картинку-палитру с подписью HEX под каждым свотчем."""
    sw = 220                 # ширина одного свотча
    sh = 180                 # высота свотча (сам прямоугольник)
    gap = 10                 # зазор
    text_h = 36              # место под текст
    width = sw * len(palette) + gap * (len(palette) + 1)
    height = sh + text_h + gap * 2

    im = Image.new("RGB", (width, height), (245, 245, 245))
    drw = ImageDraw.Draw(im)
    font = ImageFont.load_default()

    x = gap
    for rgb in palette:
        # свотч
        drw.rectangle([x, gap, x + sw, gap + sh], fill=rgb)
        # подпись
        hex_code = _hex(rgb)
        text = hex_code
        # Pillow 10+ — используем textbbox для метрик
        bbox = drw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x + (sw - tw) // 2
        ty = gap + sh + (text_h - th) // 2
        # рамка белая под текстом для контраста
        drw.rectangle([x, gap + sh, x + sw, gap + sh + text_h], fill=(255, 255, 255))
        drw.text((tx, ty), text, fill=(30, 30, 30), font=font)
        x += sw + gap

    return im

async def make_palette_photo(bot: Bot, message: types.Message, file_id: str):
    """Скачивает фото, строит палитру и отправляет изображением-ответом."""
    # скачиваем в память
    buf = io.BytesIO()
    await bot.download(file_id, buf)
    buf.seek(0)
    img = Image.open(buf)

    # считаем палитру
    palette = extract_palette(img, PALETTE_SIZE)
    palette_img = draw_palette(palette)

    # упаковываем обратно
    out = io.BytesIO()
    palette_img.save(out, format="PNG")
    out.seek(0)

    caption_lines = [f"Палитра из {PALETTE_SIZE} цветов:"]
    for rgb in palette:
        caption_lines.append(_hex(rgb))
    caption = "\n".join(caption_lines)

    await message.answer_photo(
        photo=types.BufferedInputFile(out.read(), filename="palette.png"),
        caption=caption
    )

# -------------------
# Хэндлеры
# -------------------

async def on_start(message: types.Message):
    text = (
        "Привет! Я анализирую изображения и вытаскиваю доминирующие цвета.\n\n"
        f"Добавьте меня админом в канал @{CHANNEL_USERNAME}, публикуйте фото — "
        "я пришлю палитру в ответ к посту."
    )
    await message.answer(text)

async def on_private_photo(message: types.Message, bot: Bot):
    # берём самый большой вариант фото
    file_id = message.photo[-1].file_id
    await make_palette_photo(bot, message, file_id)

async def on_channel_photo(channel_post: types.Message, bot: Bot):
    # отвечаем в том же канале, reply на пост
    file_id = channel_post.photo[-1].file_id
    await make_palette_photo(bot, channel_post, file_id)

# --------- запуск ---------

def _token() -> str:
    t = os.getenv("TELEGRAM_BOT_TOKEN")
    if not t:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не найден в переменных окружения.")
    return t

async def run():
    bot = Bot(token=_token(), parse_mode=ParseMode.HTML)
    dp = Dispatcher()

    # Регистрация хэндлеров
    dp.message.register(on_start, CommandStart())
    dp.message.register(on_private_photo, F.photo)       # фото в ЛС с ботом
    dp.channel_post.register(on_channel_photo, F.photo)  # фото в канале (бот админ)

    # Страховка от конфликтов: убираем webhook и хвосты
    try:
        await bot.delete_webhook(drop_pending_updates=True)
    except Exception as e:
        logging.warning("delete_webhook warning: %s", e)

    # Главный цикл polling c обработкой 409 и Unauthorized
    while True:
        try:
            logging.info("Start polling")
            await dp.start_polling(
                bot,
                allowed_updates=dp.resolve_used_update_types(),
                handle_signals=False,
            )
        except TelegramConflictError:
            # Другой инстанс опрашивает getUpdates. Ждём и пробуем снова.
            wait = round(random.uniform(1.5, 3.5), 2)
            logging.warning("409 Conflict. Похоже, второй инстанс. Сплю %.2f сек и пробую снова…", wait)
            await asyncio.sleep(wait)
        except TelegramUnauthorizedError:
            logging.error("Unauthorized: проверь токен бота (TELEGRAM_BOT_TOKEN).")
            await asyncio.sleep(5)
        except Exception as e:
            logging.exception("Unhandled error in polling: %s", e)
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(run())
