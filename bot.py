import os
import io
import asyncio
import logging
import tempfile
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.client.default import DefaultBotProperties
from aiogram.types import FSInputFile

# ─── SETTINGS ──────────────────────────────────────────────────────────────────
# Токен берём ТОЛЬКО из переменных окружения на Render
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN".lower())

# Канал: я подставил твои данные, их можно переопределить переменными окружения.
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "assistantdesign")          # без @
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "-10026082781747"))                 # именно отрицательное

# Требовать подписку в ЛС? (True/False)
REQUIRE_SUB = False

# Количество цветов в палитре
PALETTE_SIZE = 5

# ─── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("palette-bot")

# ─── COLOR UTILS ───────────────────────────────────────────────────────────────
def extract_palette(img: Image.Image, k: int = PALETTE_SIZE) -> List[Tuple[int,int,int]]:
    # уменьшим для скорости
    img_small = img.copy()
    img_small.thumbnail((400, 400))
    arr = np.asarray(img_small.convert("RGB")).reshape(-1, 3).astype(np.float32)

    # KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(arr)
    centers = km.cluster_centers_.astype(int)

    # сортировка по частоте кластера
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    palette = [tuple(map(int, centers[i])) for i in order]
    return palette

def rgb_to_hex(rgb: Tuple[int,int,int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def draw_palette(palette: List[Tuple[int,int,int]]) -> Image.Image:
    sw = 180   # ширина свача
    sh = 120   # высота свача
    pad = 20   # поля
    text_h = 26

    width = pad*2 + sw*len(palette)
    height = pad*2 + sh + text_h

    img = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, rgb in enumerate(palette):
        x = pad + i*sw
        y = pad
        draw.rectangle([x, y, x+sw-1, y+sh-1], fill=rgb)
        hex_code = rgb_to_hex(rgb)
        w, h = draw.textsize(hex_code, font=font)
        draw.text((x + (sw-w)//2, y + sh + 4), hex_code, fill=(10,10,10), font=font)

    return img

# ─── BOT ───────────────────────────────────────────────────────────────────────
dp = Dispatcher()

@dp.message(CommandStart())
async def cmd_start(message: types.Message, bot: Bot):
    text = (
        "Привет! Я анализирую изображения и вытаскиваю доминирующие цвета.\n\n"
        f"Добавь меня *админом* в канал @{CHANNEL_USERNAME}, опубликуй фото — "
        "я пришлю палитру в ответ к посту.\n\n"
        "Также можно прислать фото в этот чат."
    )
    await message.answer(text, parse_mode=ParseMode.MARKDOWN)

async def _download_largest_photo(bot: Bot, message: types.Message) -> str:
    """Скачиваем самое большое превью фото в temp-файл, возвращаем путь."""
    ph = message.photo[-1]  # самое большое превью
    file = await bot.get_file(ph.file_id)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    await bot.download(file, destination=tmp.name)
    return tmp.name

async def _process_and_send_palette(bot: Bot, target_chat_id: int, reply_to: int, image_path: str):
    """Строим палитру и отправляем в target_chat_id, отвечая на сообщение reply_to."""
    try:
        with Image.open(image_path) as im:
            palette = extract_palette(im, PALETTE_SIZE)
    except Exception as e:
        log.exception("Ошибка извлечения палитры: %s", e)
        await bot.send_message(target_chat_id, "Не удалось обработать изображение 😔", reply_to_message_id=reply_to)
        return
    finally:
        # удаляем временный файл
        try:
            os.unlink(image_path)
        except Exception:
            pass

    # рендер картинки палитры
    pal_img = draw_palette(palette)
    buf = io.BytesIO()
    pal_img.save(buf, format="PNG")
    buf.seek(0)

    # подпись с HEX
    hex_lines = [rgb_to_hex(c) for c in palette]
    caption = "Палитра:\n" + " ".join(hex_lines)

    # отправка
    await bot.send_photo(
        chat_id=target_chat_id,
        photo=buf,
        caption=caption,
        reply_to_message_id=reply_to
    )

# ── ЛС: пользователь прислал фото ─────────────────────────────────────────────
@dp.message(F.photo & F.chat.type == "private")
async def on_private_photo(message: types.Message, bot: Bot):
    log.info("ЛС: пришло фото от %s", message.from_user.id)

    # по желанию можно требовать подписку на канал
    if REQUIRE_SUB:
        try:
            member = await bot.get_chat_member(CHANNEL_ID, message.from_user.id)
            if member.status not in ("member", "administrator", "creator"):
                await message.answer(f"Для использования подпишись на @{CHANNEL_USERNAME} 🙂")
                return
        except Exception:
            await message.answer(f"Для использования подпишись на @{CHANNEL_USERNAME} 🙂")
            return

    tmp_path = await _download_largest_photo(bot, message)
    await _process_and_send_palette(bot, message.chat.id, message.message_id, tmp_path)

# ── КАНАЛ: новый пост с фото ──────────────────────────────────────────────────
@dp.channel_post(F.photo)
async def on_channel_photo(message: types.Message, bot: Bot):
    # Подстрахуемся: реагируем только на нужный канал,
    # если бот админ этого канала.
    if message.chat.id != CHANNEL_ID:
        log.info("Фото из другого канала (%s) проигнорировано", message.chat.id)
        return

    log.info("Канал: получено фото в %s (msg_id=%s)", message.chat.id, message.message_id)
    tmp_path = await _download_largest_photo(bot, message)
    await _process_and_send_palette(bot, message.chat.id, message.message_id, tmp_path)

# ── ТЕХНИЧЕСКОЕ ───────────────────────────────────────────────────────────────
async def main():
    if not BOT_TOKEN:
        raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN в переменных окружения.")

    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    log.info("Бот запущен. Канал: @%s (id=%s)", CHANNEL_USERNAME, CHANNEL_ID)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())

if __name__ == "__main__":
    asyncio.run(main())
