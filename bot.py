# bot.py
import os
import io
import math
import logging
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton,
    BufferedInputFile,
)

from sklearn.cluster import KMeans

# ---------- CONFIG ----------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # НЕ хардкодим
CHANNEL_USERNAME = "@desbalances"            # канал-подписка (публичный)
PALETTE_COLS = 3
PALETTE_ROWS = 4
PALETTE_SIZE = (900, 1200)  # ширина, высота итоговой карточки
BORDER = 24                 # внешний отступ
GAP = 18                    # зазор между плитками
CAPTION_HTML = "Палитра: "  # префикс к подписи
# ----------------------------

if not BOT_TOKEN:
    raise RuntimeError("Нет TELEGRAM_BOT_TOKEN в переменных окружения.")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("color-bot")

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode="HTML")
)
dp = Dispatcher()
rt = Router()
dp.include_router(rt)


# ===== Utils: цветовые преобразования =====
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)


def _rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    # rgb: (..., 3) float in [0,255]
    r, g, b = [_srgb_to_linear(rgb[..., i]) for i in range(3)]
    # матрица для D65
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    return np.stack([x, y, z], axis=-1)


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    # опорные белые D65
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = xyz[..., 0] / Xn, xyz[..., 1] / Yn, xyz[..., 2] / Zn

    def f(t):
        e = 216 / 24389
        k = 24389 / 27
        return np.where(t > e, np.cbrt(t), (k * t + 16) / 116)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return _xyz_to_lab(_rgb_to_xyz(rgb.astype(np.float32)))


def color_distance_lab(c1: np.ndarray, c2: np.ndarray) -> float:
    return float(np.linalg.norm(c1 - c2))


def hex_of(rgb: np.ndarray) -> str:
    r, g, b = [int(max(0, min(255, round(x)))) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


# ===== Алгоритм палитры =====
def extract_palette(
    pil_img: Image.Image,
    n_colors: int = 12,
    kmeans_buckets: int = 18,
    sample_max: int = 120000
) -> List[Tuple[np.ndarray, str]]:
    """
    1) ресайз и сэмплинг,
    2) удаляем экстремумы яркости по квантилям,
    3) KMeans -> центры,
    4) выбираем 12 цветов: первый — ближе к медианной L*, дальше жадно добавляем
       максимизируя минимальную дистанцию по Lab (разнообразие).
    """
    img = pil_img.convert("RGB")
    # мягкий ресайз до ~1Мп, чтобы и качество, и скорость
    w, h = img.size
    scale = (1024 * 1024 / max(1, w * h)) ** 0.5
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.asarray(img, dtype=np.uint8)
    flat = arr.reshape(-1, 3)
    # subsample
    if flat.shape[0] > sample_max:
        idx = np.random.choice(flat.shape[0], sample_max, replace=False)
        flat = flat[idx]

    # отсекаем слишком тёмные/светлые по V (HSV) квантилями
    # простая V: max(rgb)/255
    v = np.max(flat, axis=1).astype(np.float32) / 255.0
    lo, hi = np.quantile(v, [0.04, 0.96])
    mask = (v >= lo) & (v <= hi)
    base = flat[mask]
    if base.shape[0] < n_colors * 5:
        base = flat  # fallback если фото очень контрастное

    # KMeans для предварительных центров (без тяжёлых init-итераций)
    km = KMeans(n_clusters=min(kmeans_buckets, len(base)), n_init=3, random_state=42)
    km.fit(base.astype(np.float32))
    centers = km.cluster_centers_.astype(np.float32)

    # Переведём в Lab для метрики
    labs = rgb_to_lab(centers)

    # 1-й цвет — ближе к медианному L*
    Ls = labs[:, 0]
    median_L = np.median(Ls)
    first_idx = int(np.argmin(np.abs(Ls - median_L)))

    selected = [first_idx]
    selected_lab = [labs[first_idx]]

    # дальше — greedy max-min (разнообразие)
    while len(selected) < min(n_colors, len(centers)):
        dists = []
        for i in range(len(centers)):
            if i in selected:
                dists.append(-1.0)
                continue
            # мин. расстояние до уже выбранных
            md = min(color_distance_lab(labs[i], s) for s in selected_lab)
            dists.append(md)
        next_idx = int(np.argmax(dists))
        if dists[next_idx] <= 0:
            break
        selected.append(next_idx)
        selected_lab.append(labs[next_idx])

    # итоговые RGB и HEX
    result_rgbs = [centers[i] for i in selected]
    hexes = [hex_of(c) for c in result_rgbs]
    return list(zip(result_rgbs, hexes))


def render_palette_card(colors: List[Tuple[np.ndarray, str]]) -> bytes:
    cols, rows = PALETTE_COLS, PALETTE_ROWS
    W, H = PALETTE_SIZE
    card = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(card)

    cell_w = (W - 2 * BORDER - (cols - 1) * GAP) // cols
    cell_h = (H - 2 * BORDER - (rows - 1) * GAP) // rows

    # Подписи шрифтом по умолчанию (на сервере нет ttf)
    font = ImageFont.load_default()

    # рисуем плитки
    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= len(colors):
                break
            rgb, hx = colors[i]
            x0 = BORDER + c * (cell_w + GAP)
            y0 = BORDER + r * (cell_h + GAP)
            # прямоугольник цвета
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h - 24], fill=tuple(map(int, rgb)))
            # белая плашка под подпись
            draw.rectangle([x0, y0 + cell_h - 24, x0 + cell_w, y0 + cell_h], fill="white")
            # подпись
            tw, th = draw.textsize(hx, font=font)
            draw.text(
                (x0 + (cell_w - tw) // 2, y0 + cell_h - 20),
                hx,
                fill="black",
                font=font,
            )
            i += 1

    bio = io.BytesIO()
    card.save(bio, format="PNG", optimize=True)
    return bio.getvalue()


# ===== Проверка подписки =====
async def is_subscribed(user_id: int) -> bool:
    try:
        m = await bot.get_chat_member(CHANNEL_USERNAME, user_id)
        return m.status in ("member", "creator", "administrator")
    except Exception as e:
        log.warning("get_chat_member failed: %s", e)
        # Если канал недоступен, безопаснее не пускать
        return False


def subscribe_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Подписаться"), KeyboardButton(text="Проверить подписку")],
        ],
        resize_keyboard=True
    )


# ===== Handlers =====
WELCOME_TEXT = (
    "Привет! Я —  генератор цветов от ДИЗ БАЛАНС 🎨 "
    "Отправь мне фото, а я тебе отправлю его цветовую палитру в ответ."
)

@rt.message(CommandStart())
async def on_start(message: Message):
    user_id = message.from_user.id
    if await is_subscribed(user_id):
        await message.answer(WELCOME_TEXT)
    else:
        await message.answer(
            "Доступ только для подписчиков канала.\n"
            "Пожалуйста, подпишитесь и нажмите «Проверить подписку».",
            reply_markup=subscribe_keyboard(),
        )


@rt.message(F.text.lower().in_({"подписаться"}))
async def on_subscribe_button(message: Message):
    await message.answer("Откройте канал и подпишитесь: https://t.me/desbalances")


@rt.message(F.text.lower().in_({"проверить подписку"}))
@rt.message(Command("check"))
async def on_check(message: Message):
    if await is_subscribed(message.from_user.id):
        await message.answer("Спасибо за подписку! Пришлите фото — сделаю палитру.")
    else:
        await message.answer("Пока подписка не найдена. Подпишитесь и попробуйте снова.")


@rt.message(F.photo)
async def on_photo(message: Message):
    user_id = message.from_user.id
    if not await is_subscribed(user_id):
        await message.answer(
            "Доступ только для подписчиков канала.\n"
            "Подпишитесь и нажмите «Проверить подписку».",
            reply_markup=subscribe_keyboard(),
        )
        return

    # самая большая версия фото
    ph = message.photo[-1]
    try:
        # качаем в память
        file = await bot.get_file(ph.file_id)
        buf = io.BytesIO()
        await bot.download(file, buf)
        buf.seek(0)

        img = Image.open(buf).convert("RGB")
        colors = extract_palette(img, n_colors=12, kmeans_buckets=20)
        png_bytes = render_palette_card(colors)

        # подпись
        caption = CAPTION_HTML + " ".join(h for _, h in colors)

        # aiogram 3: обязательно через BufferedInputFile
        photo_input = BufferedInputFile(png_bytes, filename="palette.png")
        await message.answer_photo(photo=photo_input, caption=caption)

    except Exception as e:
        log.exception("process failed")
        await message.answer(
            f"Ошибка обработки: {e.__class__.__name__}. "
            "Попробуйте другое фото или ещё раз."
        )


async def main():
    log.info("color-bot | Бот запущен. Канал: %s", CHANNEL_USERNAME)
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
