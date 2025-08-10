#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import asyncio
import logging
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, InputFile

# ------------------ НАСТРОЙКИ ------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()  # токен берём только из ENV!
if not BOT_TOKEN:
    raise SystemExit("❗ Переменная окружения BOT_TOKEN не задана")

CHANNEL_ID = -1002608787147                  # <-- твой канал (можно светить)
CHANNEL_USERNAME = "assistantdesign"         # <-- username без @

COLORS_COUNT = 12        # 3×4
GRID_COLS = 3
GRID_ROWS = 4
HEX_FONT_SIZE = 58       # «оптимальный» крупный размер
PADDING = 36             # внешние отступы и «шаг» сетки

GREETING = (
    "Привет! Я —  генератор цветов от ДИЗ БАЛАНС 🎨 "
    "Отправь мне фото, а я тебе отправлю его цветовую палитру в ответ."
)
ASK_PDF = "Сохранить как PDF?"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ColorPaletteBOT")

bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()


# ------------------ ШРИФТ ------------------
def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/System/Library/Fonts/Supplemental/Arial.ttf",     # macOS
        "/Library/Fonts/Arial.ttf",                         # macOS
        "C:\\Windows\\Fonts\\arial.ttf",                    # Windows
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ------------------ ЦВЕТА / ПАЛИТРА ------------------
def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def sort_colors_by_hue(colors: np.ndarray) -> np.ndarray:
    import colorsys
    hsv = [colorsys.rgb_to_hsv(c[0]/255, c[1]/255, c[2]/255) for c in colors]
    idx = np.lexsort((np.array([s for h,s,v in hsv]), np.array([h for h,s,v in hsv])))
    return colors[idx]


def extract_palette(img: Image.Image, n_colors: int = 12) -> np.ndarray:
    """Извлекаем n_colors доминантов KMeans и сортируем по оттенку (без «смягчения к белому»)."""
    base = img.convert("RGB")
    base.thumbnail((640, 640), Image.LANCZOS)
    arr = np.asarray(base, dtype=np.uint8).reshape(-1, 3)

    if arr.shape[0] > 120_000:
        sel = np.random.choice(arr.shape[0], 120_000, replace=False)
        arr = arr[sel]

    km = KMeans(n_clusters=n_colors, n_init=6, random_state=42)
    km.fit(arr)
    centers = np.clip(km.cluster_centers_.round().astype(np.uint8), 0, 255)
    return sort_colors_by_hue(centers)


def draw_palette_grid(colors: np.ndarray, cols=3, rows=4) -> Image.Image:
    """Карточка 3×4: большой свотч + крупный HEX под ним. Без списков внизу."""
    assert len(colors) >= cols*rows
    card_w = 320
    swatch_h = 200
    label_h = 100

    W = PADDING*2 + cols*card_w + (cols-1)*PADDING
    H = PADDING*2 + rows*(swatch_h + label_h) + (rows-1)*PADDING

    im = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(im)
    font = load_font(HEX_FONT_SIZE)

    k = 0
    for r in range(rows):
        for c in range(cols):
            if k >= len(colors):
                break
            x = PADDING + c*(card_w + PADDING)
            y = PADDING + r*(swatch_h + label_h + PADDING)

            color = tuple(int(v) for v in colors[k])
            # свотч
            draw.rectangle([x, y, x + card_w, y + swatch_h], fill=color)

            # подпись HEX
            hex_code = rgb_to_hex(color)
            tw, th = draw.textbbox((0, 0), hex_code, font=font)[2:]
            tx = x + (card_w - tw)//2
            ty = y + swatch_h + (label_h - th)//2
            draw.text((tx, ty), hex_code, fill=(28, 54, 54), font=font)
            k += 1

    return im


def make_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def make_pdf_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PDF")
    return buf.getvalue()


# Кэш последних карточек по чату/пользователю (для /pdf)
LAST_CARD_IMG: dict[int, Image.Image] = {}


# ------------------ ПРОВЕРКА ПОДПИСКИ ------------------
async def is_subscriber(user_id: int) -> bool:
    try:
        m = await bot.get_chat_member(CHANNEL_ID, user_id)
        return m.status in ("member", "administrator", "creator")
    except Exception as e:
        log.warning(f"subscription check failed: {e}")
        # если не смогли проверить — запрещаем
        return False


# ------------------ ХЕНДЛЕРЫ ------------------
@dp.message(CommandStart())
async def on_start(message: types.Message):
    await message.answer(GREETING)


@dp.message(Command("pdf"))
async def on_pdf(message: types.Message):
    img = LAST_CARD_IMG.get(message.chat.id) or LAST_CARD_IMG.get(message.from_user.id)
    if not img:
        await message.answer("Нет последней палитры для сохранения.")
        return
    pdf = make_pdf_bytes(img)
    await message.answer_document(document=InputFile(io.BytesIO(pdf), filename="palette.pdf"),
                                  caption="Сохранено как PDF.")


@dp.callback_query()
async def on_cb(call: types.CallbackQuery):
    if call.data == "save_pdf":
        img = LAST_CARD_IMG.get(call.message.chat.id) or LAST_CARD_IMG.get(call.from_user.id)
        if not img:
            await call.answer("Нет последней палитры", show_alert=True)
            return
        pdf = make_pdf_bytes(img)
        await call.message.reply_document(document=InputFile(io.BytesIO(pdf), filename="palette.pdf"),
                                          caption="Сохранено как PDF.")
        await call.answer("Готово ✅")
    else:
        await call.answer()


@dp.message(F.photo & (F.chat.type == "private"))
async def on_private_photo(message: types.Message):
    # доступ только подписчикам канала
    if not await is_subscriber(message.from_user.id):
        await message.reply(f"Этот бот доступен только подписчикам канала @{CHANNEL_USERNAME}.")
        return

    # скачать фото
    file = await bot.get_file(message.photo[-1].file_id)
    bio = io.BytesIO()
    await bot.download_file(file.file_path, bio)
    bio.seek(0)
    img = Image.open(bio).convert("RGB")

    # палитра
    colors = extract_palette(img, COLORS_COUNT)
    card = draw_palette_grid(colors, GRID_COLS, GRID_ROWS)
    LAST_CARD_IMG[message.from_user.id] = card
    LAST_CARD_IMG[message.chat.id] = card  # на всякий

    png = make_png_bytes(card)
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Сохранить как PDF", callback_data="save_pdf")]
    ])
    await message.reply_photo(photo=png, caption=ASK_PDF, reply_markup=kb)


@dp.channel_post(F.photo)  # посты в самом канале
async def on_channel_photo(message: types.Message):
    # отвечаем только для нашего канала (на всякий случай)
    if message.chat.id != CHANNEL_ID:
        return

    file = await bot.get_file(message.photo[-1].file_id)
    bio = io.BytesIO()
    await bot.download_file(file.file_path, bio)
    bio.seek(0)
    img = Image.open(bio).convert("RGB")

    colors = extract_palette(img, COLORS_COUNT)
    card = draw_palette_grid(colors, GRID_COLS, GRID_ROWS)
    LAST_CARD_IMG[message.chat.id] = card

    png = make_png_bytes(card)
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Сохранить как PDF", callback_data="save_pdf")]
    ])
    # отвечаем реплаем на пост
    await bot.send_photo(chat_id=message.chat.id, photo=png,
                         caption=ASK_PDF,
                         reply_to_message_id=message.message_id,
                         reply_markup=kb)


# ------------------ ЗАПУСК (один, без 409) ------------------
async def _delete_webhook():
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        log.info("Webhook removed (drop_pending_updates=True)")
    except Exception as e:
        log.warning(f"delete_webhook warning: {e}")

async def main():
    await _delete_webhook()
    await dp.start_polling(bot, allowed_updates=["message", "channel_post", "callback_query"])

if __name__ == "__main__":
    asyncio.run(main())
