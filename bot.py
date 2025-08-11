import asyncio
import os
from io import BytesIO
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import (
    Message, BufferedInputFile, InlineKeyboardMarkup, InlineKeyboardButton,
    ChatMemberLeft, ChatMemberBanned
)

# === Конфиг ===
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHANNEL_USERNAME = os.getenv("MAIN_CHANNEL_USERNAME", "desbalances")  # без @
CHANNEL_LINK = f"https://t.me/{CHANNEL_USERNAME}"

assert BOT_TOKEN, "Env TELEGRAM_BOT_TOKEN is required"

bot = Bot(BOT_TOKEN)
dp = Dispatcher()

# Кэшируем id канала, чтобы уметь отвечать на посты канала
_channel_id_cache: int | None = None


async def get_channel_id() -> int:
    global _channel_id_cache
    if _channel_id_cache is None:
        chat = await bot.get_chat(f"@{CHANNEL_USERNAME}")
        _channel_id_cache = chat.id
    return _channel_id_cache


# === Утилиты подписки ===
async def is_subscriber(user_id: int) -> bool:
    """
    Возвращает True, если user_id подписан на канал.
    """
    try:
        member = await bot.get_chat_member(f"@{CHANNEL_USERNAME}", user_id)
        if isinstance(member, (ChatMemberLeft, ChatMemberBanned)):
            return False
        return True
    except Exception:
        # Закрытые аккаунты/ограничения — считаем не подписан
        return False


def subscribe_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="📌 Подписаться", url=CHANNEL_LINK),
        InlineKeyboardButton(text="🔄 Проверить подписку", callback_data="check_sub")
    ]])


# === Генерация палитры ===

def _hex(c: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*c)

def _prepare_pixels(im: Image.Image, sample: int = 160_000) -> np.ndarray:
    # уменьшаем до ~512px по длинной стороне, конвертим к RGB
    im = im.convert("RGB")
    w, h = im.size
    scale = 512 / max(w, h) if max(w, h) > 512 else 1.0
    if scale < 1:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.asarray(im).reshape(-1, 3).astype(np.float32)

    # легкая фильтрация «шумов»: убираем почти-чисто белые/черные пиксели с малой вероятностью
    # (не жестко, чтобы не потерять светлые/темные оттенки)
    brightness = arr.mean(axis=1)
    mask = np.ones(len(arr), dtype=bool)
    mask &= ~((brightness < 5) | (brightness > 250))
    arr = arr[mask]

    # случайная подвыборка для скорости
    if len(arr) > sample:
        idx = np.random.choice(len(arr), sample, replace=False)
        arr = arr[idx]
    return arr

def _kmeans_colors(pixels: np.ndarray, k: int) -> List[Tuple[int, int, int]]:
    # KMeans с учетом частоты: центры в RGB
    # sklearn>=1.2: n_init='auto' допустимо; для совместимости укажем конкретное число.
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_.astype(int)

    # считаем частоты кластеров и сортируем по убыванию площади, затем по яркости
    counts = np.bincount(labels, minlength=k)
    lumin = centers.mean(axis=1)
    order = np.lexsort((lumin, -counts))  # сначала по -counts, затем по lumin
    centers = centers[order]
    return [tuple(map(int, c)) for c in centers]

def _draw_palette(colors: List[Tuple[int, int, int]], thumb: Image.Image | None) -> Image.Image:
    """
    Рисуем карточку 3x4 (12 цветов) с подписями HEX. Слева сверху — миниатюра исходного фото.
    """
    cols, rows = 4, 3
    cell_w, cell_h = 260, 150
    pad = 28
    title_h = 68
    thumb_box = 220  # миниатюра исходного

    W = pad*2 + cols*cell_w
    H = pad*3 + rows*cell_h + title_h + thumb_box

    img = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Заголовок
    title = "Палитра (12 цветов)"
    tb = draw.textbbox((0, 0), title, font=font)
    draw.text(((W - (tb[2]-tb[0]))//2, pad), title, fill=(30,30,30), font=font)

    # Миниатюра
    if thumb:
        t = thumb.convert("RGB")
        tw, th = t.size
        scale = min(thumb_box/tw, thumb_box/th, 1.0)
        t = t.resize((int(tw*scale), int(th*scale)), Image.LANCZOS)
        tx = pad
        ty = title_h + pad
        # рамка
        draw.rectangle([tx-1, ty-1, tx+t.width+1, ty+t.height+1], outline=(200,200,200), width=2)
        img.paste(t, (tx, ty))

    # Сетка цветов
    start_y = title_h + pad
    # если вставили миниатюру, отодвинем сетку вправо
    grid_offset_x = pad + thumb_box + pad if thumb else pad

    def text_centered(rect, text):
        x0, y0, x1, y1 = rect
        bbox = draw.textbbox((0,0), text, font=font)
        tx = x0 + (x1-x0 - (bbox[2]-bbox[0]))//2
        ty = y0 + (y1-y0 - (bbox[3]-bbox[1]))//2
        draw.text((tx, ty), text, fill=(30,30,30), font=font)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(colors): break
            x0 = grid_offset_x + c*cell_w
            y0 = start_y + r*cell_h
            x1 = x0 + cell_w - pad//2
            y1 = y0 + cell_h - pad//2

            # прямоугольник цвета
            color_rect = (x0+6, y0+6, x1-6, y0 + int((y1-y0)*0.6))
            draw.rectangle(color_rect, fill=colors[idx], outline=(220,220,220), width=2)

            # подпись
            hex_text = _hex(colors[idx])
            label_rect = (x0, color_rect[3]+8, x1, y1-6)
            text_centered(label_rect, hex_text)
            idx += 1

    return img


async def generate_palette_image(photo_bytes: bytes, k: int = 12) -> BytesIO:
    original = Image.open(BytesIO(photo_bytes))
    pixels = _prepare_pixels(original)
    if len(pixels) < k:
        raise RuntimeError("Недостаточно пикселей для кластеризации")

    colors = _kmeans_colors(pixels, k)
    card = _draw_palette(colors, original)
    out = BytesIO()
    card.save(out, format="PNG")
    out.seek(0)
    return out


# === Хендлеры ===

WELCOME = (
    "Привет! Я — генератор цветов от ДИЗ БАЛАНС 🎨 "
    "Отправь мне фото, а я тебе отправлю его цветовую палитру в ответ."
)

@dp.message(CommandStart())
async def on_start(message: Message):
    user_id = message.from_user.id
    if not await is_subscriber(user_id):
        await message.answer(
            "Этот инструмент доступен только подписчикам канала. "
            "Подпишись и вернись — разблокирую генератор.",
            reply_markup=subscribe_kb()
        )
        return
    await message.answer(WELCOME)

@dp.callback_query(F.data == "check_sub")
async def on_check_sub(call):
    uid = call.from_user.id
    if await is_subscriber(uid):
        await call.message.edit_text("Готово! Подписка найдена. Можешь присылать фото 🎯")
    else:
        await call.answer("Пока не вижу подписки 🙏", show_alert=True)

@dp.message(F.photo)
async def on_photo_private(message: Message):
    # если это приват-чат — проверяем подписку
    if message.chat.type == "private":
        if not await is_subscriber(message.from_user.id):
            await message.answer(
                "Инструмент доступен только подписчикам. "
                "Подпишись на канал и нажми «Проверить подписку».",
                reply_markup=subscribe_kb()
            )
            return

    # грузим файл
    try:
        file = await bot.get_file(message.photo[-1].file_id)
        photo_bytes = await bot.download_file(file.file_path)
        photo_data = photo_bytes.read()

        out = await generate_palette_image(photo_data, k=12)
        await message.reply_photo(BufferedInputFile(out.getvalue(), filename="palette.png"),
                                  caption="Готово! Палитра: 12 цветов (HEX на карточках)")
    except Exception:
        await message.reply("Не удалось обработать изображение. Попробуйте другое фото.")

# Сообщения в КАНАЛЕ: отвечаем палитрой в треде поста
@dp.channel_post(F.photo)
async def on_channel_photo(message: Message):
    try:
        # убеждаемся, что это именно наш канал
        if message.chat.id != await get_channel_id():
            return
        file = await bot.get_file(message.photo[-1].file_id)
        b = await bot.download_file(file.file_path)
        out = await generate_palette_image(b.read(), k=12)
        await message.reply_photo(BufferedInputFile(out.getvalue(), filename="palette.png"),
                                  caption="Палитра: 12 цветов")
    except Exception:
        await message.reply("Не удалось обработать изображение. Попробуйте другое фото.")

async def main():
    # Разрешим нужные апдейты, чтобы снизить шум
    await dp.start_polling(bot, allowed_updates=["message", "channel_post", "callback_query"])

if __name__ == "__main__":
    asyncio.run(main())
