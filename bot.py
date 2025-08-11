
import os
import io
import math
import asyncio
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
from aiogram import Bot, Dispatcher, F, Router, types
from aiogram.enums import ParseMode, ChatMemberStatus
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from aiogram.types.input_file import BufferedInputFile

# ========= Настройки =========
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHANNEL_USERNAME = "desbalances"  # без @
NUM_COLORS = 12
PRECLUSTERS = 18  # берем больше кластеров, потом оставляем 12 самых репрезентативных

if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в переменных окружения.")

# aiogram 3.7+ vs 3.6 fallback: включаем HTML-парсинг
try:
    from aiogram.client.default_bot_properties import DefaultBotProperties
    bot = Bot(token=BOT_TOKEN, default_bot_properties=DefaultBotProperties(parse_mode=ParseMode.HTML))
except Exception:
    bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)

dp = Dispatcher()
router = Router()
dp.include_router(router)

# ========= Утилиты =========
def pil_text_size(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, text: str) -> Tuple[int, int]:
    """Совместимо с Pillow>=10: используем textbbox вместо устаревшего textsize."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def hex_to_rgb(s: str) -> Tuple[int, int, int]:
    return ImageColor.getrgb(s)

def preprocess_image(img: Image.Image, target_max=512) -> np.ndarray:
    # приводим к умеренному размеру для устойчивости кластеризации
    w, h = img.size
    scale = min(1.0, target_max / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return arr

def kmeans_palette(arr: np.ndarray, n_pre: int, n_final: int) -> List[Tuple[int, int, int]]:
    # lazy import, чтобы быстрее стартовать
    from sklearn.cluster import KMeans

    pixels = arr.reshape(-1, 3).astype(np.float32)

    # Нормализуем яркость, чтобы не «забивало» серыми тонами
    pixels_lab = rgb_to_oklab(pixels)  # (N,3) в Oklab — лучше разносит воспринимаемые цвета

    # Кластеризация с запасом
    km = KMeans(n_clusters=n_pre, n_init="auto", init="k-means++", random_state=42)
    labels = km.fit_predict(pixels_lab)
    centers_lab = km.cluster_centers_

    # Вес кластера = сколько пикселей
    counts = np.bincount(labels, minlength=n_pre).astype(np.float32)

    # Перевод центров обратно в RGB
    centers_rgb = oklab_to_rgb(centers_lab).clip(0, 255).astype(np.uint8)

    # Отбираем разнообразную 12‑ку: жадно добавляем самый «тяжелый», затем каждый следующий — максимально далекий
    chosen = []
    chosen_idx = int(np.argmax(counts))
    chosen.append(chosen_idx)

    def dist(i, j):
        # расстояние в Oklab (воспринимаемое)
        di = centers_lab[i] - centers_lab[j]
        return float(np.sqrt(np.dot(di, di)))

    while len(chosen) < n_final and len(chosen) < len(centers_rgb):
        best = -1
        best_score = -1.0
        for i in range(len(centers_rgb)):
            if i in chosen:
                continue
            # минимальная дистанция до уже выбранных (чтобы не брать «похожий»)
            dmin = min(dist(i, j) for j in chosen)
            # смешанный критерий: разнообразие * вес
            score = dmin * (1.0 + math.log1p(counts[i]))
            if score > best_score:
                best_score = score
                best = i
        if best == -1:
            break
        chosen.append(best)

    # Если вдруг центров меньше 12 — просто берем сколько есть
    palette = [tuple(int(x) for x in centers_rgb[i]) for i in chosen[:n_final]]
    # На всякий случай: если вышло меньше — добьём наиболее «весомыми» невыбранными
    if len(palette) < n_final:
        remain = [i for i in np.argsort(-counts) if i not in chosen]
        for i in remain[: n_final - len(palette)]:
            palette.append(tuple(int(x) for x in centers_rgb[i]))
    return palette

def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """rgb [0..255] -> Oklab. Векторизовано."""
    # нормируем
    rgb = rgb / 255.0
    # линейное RGB (sRGB)
    def srgb_to_linear(c):
        a = 0.055
        return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)
    lrgb = srgb_to_linear(rgb)

    # матрица в LMS (из спецификации Oklab)
    M = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ], dtype=np.float64)
    lms = lrgb @ M.T
    lms = np.cbrt(lms)  # кубический корень

    ML = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ], dtype=np.float64)
    return lms @ ML.T

def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Oklab -> rgb [0..255]. Векторизовано."""
    # обратные матрицы
    ML_inv = np.array([
        [ 1.0,  0.3963377774,  0.2158037573],
        [ 1.0, -0.1055613458, -0.0638541728],
        [ 1.0, -0.0894841775, -1.2914855480],
    ], dtype=np.float64)
    lms = lab @ ML_inv.T
    lms = np.power(lms, 3.0)

    M_inv = np.array([
        [ 4.0767416621, -3.3077115913,  0.2309699292],
        [-1.2684380046,  2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147,  1.7076147010],
    ], dtype=np.float64)
    lrgb = lms @ M_inv.T

    def linear_to_srgb(c):
        a = 0.055
        return np.where(c <= 0.0031308, 12.92 * c, (1 + a) * (c ** (1 / 2.4)) - a)

    srgb = linear_to_srgb(lrgb).clip(0.0, 1.0)
    return (srgb * 255.0)

def render_palette_card(colors: List[Tuple[int, int, int]]) -> bytes:
    cols, rows = 4, 3  # 12 цветов = 4x3
    cell_w, cell_h = 320, 240
    pad = 16
    label_h = 56
    w = cols * cell_w + (cols + 1) * pad
    h = rows * (cell_h + label_h) + (rows + 1) * pad

    img = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Шрифт: если на сервере нет ttf — Pillow подставит дефолт
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    for idx, rgb in enumerate(colors):
        r, c = divmod(idx, cols)
        x0 = pad + c * (cell_w + pad)
        y0 = pad + r * (cell_h + label_h + pad)
        # прямоугольник цвета
        draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=rgb)
        # подпись HEX
        hx = rgb_to_hex(rgb)
        tw, th = pil_text_size(draw, font, hx)
        tx = x0 + (cell_w - tw) // 2
        ty = y0 + cell_h + (label_h - th) // 2
        # подложка под текст (легкий белый)
        draw.rectangle([x0, y0 + cell_h, x0 + cell_w, y0 + cell_h + label_h], fill=(255, 255, 255))
        draw.text((tx, ty), hx, fill=(20, 20, 20), font=font)

    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

async def is_subscriber(user_id: int) -> bool:
    try:
        chat = await bot.get_chat(f"@{CHANNEL_USERNAME}")
        cm = await bot.get_chat_member(chat.id, user_id)
        return cm.status in {
            ChatMemberStatus.MEMBER,
            ChatMemberStatus.ADMINISTRATOR,
            ChatMemberStatus.CREATOR,
            ChatMemberStatus.OWNER,  # на всякий
        }
    except Exception:
        # если телеграм не ответил — перестрахуемся и попросим подписаться/проверить
        return False

def sub_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="📌 Подписаться", url=f"https://t.me/{CHANNEL_USERNAME}"),
        InlineKeyboardButton(text="🔄 Проверить подписку", callback_data="check_sub"),
    ]])

# ========= Хэндлеры =========
WELCOME_TEXT = (
    "Привет! Я —  генератор цветов от ДИЗ БАЛАНС 🎨 "
    "Отправь мне фото, а я тебе отправлю его цветовую палитру в ответ."
)

@router.message(CommandStart())
async def on_start(message: Message):
    if await is_subscriber(message.from_user.id):
        await message.answer(WELCOME_TEXT)
    else:
        await message.answer(
            "Бот доступен только подписчикам канала. Подпишись и нажми «Проверить подписку».",
            reply_markup=sub_keyboard()
        )

@router.callback_query(F.data == "check_sub")
async def on_check_sub(cb: types.CallbackQuery):
    if await is_subscriber(cb.from_user.id):
        await cb.message.edit_text(WELCOME_TEXT)
    else:
        await cb.answer("Пока не вижу подписки. Проверь ещё раз позже.", show_alert=True)

@router.message(F.photo)
async def on_photo(message: Message):
    # защита: только подписчики
    if not await is_subscriber(message.from_user.id):
        await message.answer(
            "Бот доступен только подписчикам канала. Подпишись и нажми «Проверить подписку».",
            reply_markup=sub_keyboard()
        )
        return

    try:
        file_id = message.photo[-1].file_id
        file = await bot.get_file(file_id)
        file_bytes = await bot.download_file(file.file_path)
        img = Image.open(io.BytesIO(file_bytes.read()))

        arr = preprocess_image(img)
        colors = kmeans_palette(arr, PRECLUSTERS, NUM_COLORS)

        png = render_palette_card(colors)
        input_file = BufferedInputFile(png, filename="palette.png")
        await message.answer_photo(
            photo=input_file,
            caption="Готово! Палитра на основе фото."
        )
    except Exception as e:
        await message.answer(f"Ошибка обработки: {type(e).__name__}. Попробуйте другое фото или ещё раз.")

# ========= Точка входа =========
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
