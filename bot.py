"""
Бот: генератор палитры из 12 цветов с HEX-подписями.
Доступ к функциям в ЛС — только для подписчиков канала (@assistantdesign).
В канале бот отвечает палитрой на пост с фото (бот должен быть админом канала).

Зависимости: aiogram==3.7.0, Pillow==11.0.0, numpy==1.26.4
Команда запуска на Render (Background Worker): python bot.py
Переменные окружения: TELEGRAM_BOT_TOKEN
"""

import os
import io
import asyncio
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from aiogram.exceptions import TelegramBadRequest

# =========================
# Конфигурация
# =========================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Env var TELEGRAM_BOT_TOKEN is empty")

# Username канала (с @). Используем его для проверки подписки.
CHANNEL_USERNAME = "@assistantdesign"

WELCOME_TEXT = (
    "Привет! Я —  генератор цветов от ДИЗ БАЛАНС 🎨 "
    "Отправь мне фото, а я тебе отправлю его цветовую палитру в ответ."
)

# =========================
# Утилиты: шрифты/текст
# =========================
def load_font(size: int) -> ImageFont.ImageFont:
    """Пытаемся взять системный шрифт; если нет — дефолтный PIL."""
    for name in ("DejaVuSans.ttf", "Arial.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str,
              font: ImageFont.ImageFont) -> Tuple[int, int]:
    """Безопасно получаем размер текста (Pillow ≥10: textbbox)."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def to_hex(rgb: Tuple[int, int, int]) -> str:
    """RGB -> '#rrggbb' (нижний регистр)."""
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


# =========================
# Палитра (без scikit-learn)
# =========================
def dominant_colors(img: Image.Image, k: int = 12) -> List[Tuple[int, int, int]]:
    """
    Получаем k доминирующих цветов с помощью медианного квантайзера Pillow.
    1) уменьшаем изображение для стабильности,
    2) quantize(colors=k, method=MEDIANCUT),
    3) берём топ-k по частоте.
    """
    work = img.convert("RGB")

    # Уменьшаем, чтобы сократить шума и ускорить квантование
    max_side = 512
    scale = min(1.0, max_side / max(work.size))
    if scale < 1.0:
        work = work.resize(
            (int(work.width * scale), int(work.height * scale)),
            Image.LANCZOS
        )

    # Квантование цветов до k
    q = work.quantize(colors=k, method=Image.MEDIANCUT)
    q_rgb = q.convert("RGB")

    # Получаем список (count, (r,g,b))
    colors = q_rgb.getcolors(maxcolors=q_rgb.width * q_rgb.height) or []
    # Сортируем по убыванию count
    colors.sort(key=lambda t: t[0], reverse=True)

    # Возвращаем k RGB
    top = [tuple(map(int, rgb)) for count, rgb in colors[:k]]
    # На всякий случай добиваем до k (если вдруг меньше)
    while len(top) < k and top:
        top.append(top[len(top) % len(top)])
    return top[:k]


def build_palette_card(colors: List[Tuple[int, int, int]]) -> bytes:
    """
    Рисуем карточку 3x4 с крупными HEX‑подписями.
    Возвращаем PNG как bytes.
    """
    assert len(colors) == 12, "Ожидается ровно 12 цветов"

    cols, rows = 3, 4           # сетка 3×4
    sw, sh = 280, 220           # размер свотча
    gap, pad = 24, 24           # промежутки и поля
    label_h = 56                # высота под подписью
    bg = (245, 245, 245)        # фон карточки

    W = pad * 2 + cols * sw + (cols - 1) * gap
    H = pad * 2 + rows * (sh + label_h) + (rows - 1) * gap

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    font = load_font(28)

    for i, rgb in enumerate(colors):
        r_idx, c_idx = divmod(i, cols)  # строка/колонка
        x0 = pad + c_idx * (sw + gap)
        y0 = pad + r_idx * (sh + label_h + gap)

        # Прямоугольник цвета (со скруглением)
        draw.rounded_rectangle([x0, y0, x0 + sw, y0 + sh],
                               radius=16, fill=rgb)

        # Подпись HEX (по центру под свотчем)
        hex_code = to_hex(rgb)
        tw, th = text_size(draw, hex_code, font)
        tx = x0 + (sw - tw) // 2
        ty = y0 + sh + (label_h - th) // 2

        # Светлая плашка под текстом для читаемости
        draw.rounded_rectangle([x0 + 12, y0 + sh + 8, x0 + sw - 12,
                                y0 + sh + label_h - 8],
                               radius=12, fill=(255, 255, 255))
        draw.text((tx, ty), hex_code, fill=(40, 40, 40), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()


# =========================
# Проверка подписки
# =========================
def subscribe_keyboard() -> types.InlineKeyboardMarkup:
    """Кнопки: Подписаться / Проверить подписку."""
    return types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(
            text="📌 Подписаться",
            url=f"https://t.me/{CHANNEL_USERNAME.lstrip('@')}"
        )],
        [types.InlineKeyboardButton(
            text="🔁 Проверить подписку",
            callback_data="check_sub"
        )]
    ])


async def is_subscriber(bot: Bot, user_id: int) -> bool:
    """
    Проверяем, подписан ли пользователь на канал.
    Требуется, чтобы бот видел участников (лучше назначить его админом канала).
    """
    try:
        m = await bot.get_chat_member(CHANNEL_USERNAME, user_id)
        return getattr(m, "status", "") in ("creator", "administrator", "member")
    except TelegramBadRequest:
        # Если бот не видит пользователя (канал приватный/нет прав) — считаем не подписан
        return False
    except Exception:
        return False


# =========================
# Вспомогательные действия
# =========================
async def read_input_image(bot: Bot, message: types.Message) -> Image.Image | None:
    """Скачиваем самое большое превью фото и открываем как PIL.Image."""
    try:
        # Берём самую большую версию фото
        ph = max(message.photo, key=lambda p: p.file_size or 0)
        file = await bot.get_file(ph.file_id)
        buf = io.BytesIO()
        await bot.download(file, destination=buf)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    except Exception:
        return None


async def process_and_reply(bot: Bot, message: types.Message) -> None:
    """Строим палитру 12 цветов и отвечаем картинкой + списком HEX."""
    pil = await read_input_image(bot, message)
    if pil is None:
        await message.reply("Не удалось прочитать изображение. Попробуйте другое фото.")
        return

    try:
        colors = dominant_colors(pil, k=12)              # 12 доминирующих цветов
        png = build_palette_card(colors)                 # PNG‑карточка
        hex_list = " ".join(to_hex(c) for c in colors)   # подпись (строкой)

        await message.reply_photo(
            photo=types.BufferedInputFile(png, filename="palette.png"),
            caption=f"Палитра: {hex_list}"
        )
    except Exception:
        await message.reply("Не удалось обработать изображение. Попробуйте другое фото.")


# =========================
# Хэндлеры
# =========================
async def on_start(message: types.Message, bot: Bot) -> None:
    """Приветствие: только для подписчиков."""
    user_id = message.from_user.id
    if await is_subscriber(bot, user_id):
        await message.answer(WELCOME_TEXT)
    else:
        await message.answer(
            "Этот инструмент доступен только подписчикам канала.\n"
            "Подпишись и нажми «Проверить подписку».",
            reply_markup=subscribe_keyboard()
        )


async def on_check_sub(call: types.CallbackQuery, bot: Bot) -> None:
    """Кнопка «Проверить подписку»."""
    ok = await is_subscriber(bot, call.from_user.id)
    if ok:
        await call.message.edit_text(WELCOME_TEXT)
    else:
        await call.answer("Подписка не найдена 😕", show_alert=True)


async def on_private_photo(message: types.Message, bot: Bot) -> None:
    """Фото в ЛС: пропускаем только подписчиков."""
    if not await is_subscriber(bot, message.from_user.id):
        await message.answer(
            "Доступ только для подписчиков канала.",
            reply_markup=subscribe_keyboard()
        )
        return
    await process_and_reply(bot, message)


async def on_channel_photo(channel_post: types.Message, bot: Bot) -> None:
    """
    Пост с фото в канале: бот отвечает палитрой в треде поста.
    Бот должен быть админом канала.
    """
    try:
        pil = await read_input_image(bot, channel_post)
        if pil is None:
            raise ValueError("no image")

        colors = dominant_colors(pil, k=12)
        png = build_palette_card(colors)
        hex_list = " ".join(to_hex(c) for c in colors)

        await bot.send_photo(
            chat_id=channel_post.chat.id,
            photo=types.BufferedInputFile(png, filename="palette.png"),
            caption=f"Палитра: {hex_list}",
            reply_to_message_id=channel_post.message_id
        )
    except Exception:
        await bot.send_message(
            chat_id=channel_post.chat.id,
            text="Не удалось обработать изображение. Попробуйте другое фото.",
            reply_to_message_id=channel_post.message_id
        )


# =========================
# Точка входа
# =========================
async def main() -> None:
    """Запуск long polling."""
    bot = Bot(token=BOT_TOKEN)       # без parse_mode и без DefaultBotProperties
    dp = Dispatcher()

    # Регистрация хэндлеров
    dp.message.register(on_start, CommandStart())
    dp.callback_query.register(on_check_sub, F.data == "check_sub")
    dp.message.register(on_private_photo, F.photo)
    dp.channel_post.register(on_channel_photo, F.photo)

    print("color-bot | Бот запущен. Ждём апдейты…")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
