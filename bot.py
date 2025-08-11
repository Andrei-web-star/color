import io
import os
import math
import random
from typing import List, Tuple

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.client.default_bot_properties import DefaultBotProperties
from aiogram.types import Message, BufferedInputFile
from aiogram.filters import Command

from PIL import Image, ImageDraw, ImageFont
import numpy as np


# ── настройки ──────────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # на Render переменная окружения
NUM_COLORS = 12                               # хотим всегда 12 цветов
MAX_SIDE = 512                                # до какого размера сжимать картинку
SEED = 13                                     # детерминированность k-means

WELCOME = (
    "Привет! Я —  генератор цветов от ДИЗ БАЛАНС 🎨 "
    "Отправь мне фото, а я тебе отправлю его цветовую палитру в ответ."
)


# ── утилиты цвета ──────────────────────────────────────────────────────────────
def rgb_to_hsv(c: np.ndarray) -> np.ndarray:
    """RGB [0..255] -> HSV [H 0..360, S 0..1, V 0..1] для массива (N,3)."""
    c = c.astype(np.float32) / 255.0
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    mx = np.max(c, axis=1)
    mn = np.min(c, axis=1)
    diff = mx - mn + 1e-6

    h = np.zeros_like(mx)
    mask = mx == r
    h[mask] = (60 * ((g[mask] - b[mask]) / diff[mask]) + 360) % 360
    mask = mx == g
    h[mask] = (60 * ((b[mask] - r[mask]) / diff[mask]) + 120) % 360
    mask = mx == b
    h[mask] = (60 * ((r[mask] - g[mask]) / diff[mask]) + 240) % 360

    s = diff / (mx + 1e-6)
    v = mx
    return np.stack([h, s, v], axis=1)


def hsv_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Эвклидова метрика в HSV с учётом круговой оси Hue."""
    dh = np.minimum(np.abs(a[0] - b[0]), 360 - np.abs(a[0] - b[0])) / 180.0  # [0..1]
    ds = np.abs(a[1] - b[1])
    dv = np.abs(a[2] - b[2])
    # веса: hue важнее, чем S/V
    return math.sqrt((1.6 * dh) ** 2 + (1.0 * ds) ** 2 + (1.0 * dv) ** 2)


def hex_color(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ── лёгкий K-Means без sklearn ────────────────────────────────────────────────
def kmeans(pixels: np.ndarray, k: int, iters: int = 12, seed: int = 13) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает (centroids[k,3], labels[n]) для RGB-пикселей.
    Простой MiniBatch: берём небольшие батчи для устойчивости и скорости.
    """
    rng = np.random.default_rng(seed)
    n = pixels.shape[0]
    # инициализация центров случайной подвыборкой
    centers = pixels[rng.choice(n, size=min(k, n), replace=False)].astype(np.float32)
    if centers.shape[0] < k:  # если пикселей мало — дублируем
        reps = k - centers.shape[0]
        centers = np.vstack([centers, centers[rng.choice(centers.shape[0], reps, replace=True)]])
    # основной цикл
    for _ in range(iters):
        # мини-батч
        batch_idx = rng.choice(n, size=min(5000, n), replace=False)
        batch = pixels[batch_idx].astype(np.float32)
        # принадлежности
        dists = np.sum((batch[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # (B, k)
        labels = np.argmin(dists, axis=1)
        # обновление центров
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centers[i] = batch[mask].mean(axis=0)
    # финальные метки на всех пикселях
    dists_all = np.sum((pixels[:, None, :].astype(np.float32) - centers[None, :, :]) ** 2, axis=2)
    labels_all = np.argmin(dists_all, axis=1)
    return centers.astype(np.uint8), labels_all


def ensure_diverse_palette(centers: np.ndarray, counts: np.ndarray, need: int) -> List[Tuple[int, int, int]]:
    """
    Берём больше кластеров, затем жёстко удаляем дубли по HSV-дистанции,
    и жадно набираем не менее `need` цветов, сохраняя разнообразие.
    """
    # сортируем центры по вкладу (частоте)
    idx = np.argsort(counts)[::-1]
    centers = centers[idx]
    counts = counts[idx]

    hsv = rgb_to_hsv(centers)
    picked = []
    picked_hsv = []
    min_dist = 0.22  # порог "дубликата" (эмпирически)

    for i, c in enumerate(centers):
        chsv = hsv[i]
        if not picked:
            picked.append(tuple(int(x) for x in c))
            picked_hsv.append(chsv)
            continue
        # проверяем на "слишком близко"
        if all(hsv_distance(chsv, ph) >= min_dist for ph in picked_hsv):
            picked.append(tuple(int(x) for x in c))
            picked_hsv.append(chsv)
        if len(picked) == need:
            break

    # если не хватило — добираем ближайшие «не очень разные»
    j = 0
    while len(picked) < need and j < len(centers):
        c = centers[j]
        t = tuple(int(x) for x in c)
        if t not in picked:
            picked.append(t)
        j += 1

    # если и так не хватило (крайне однотонная картинка) — дублируем/джиттерим
    while len(picked) < need:
        base = random.choice(picked)
        # лёгкий джиттер ±6
        jitter = tuple(int(max(0, min(255, base[i] + random.randint(-6, 6)))) for i in range(3))
        picked.append(jitter)

    return picked[:need]


def extract_palette(img: Image.Image, need: int = NUM_COLORS) -> List[Tuple[int, int, int]]:
    """Устойчивая палитра с разнообразием, без падений на однотонных фото."""
    # приводим к RGB и уменьшаем
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.array(img)  # (H, W, 3)
    # сэмплируем пиксели (чтобы не было слишком много)
    pixels = arr.reshape(-1, 3)
    n = pixels.shape[0]
    if n > 120_000:
        rng = np.random.default_rng(SEED)
        pixels = pixels[rng.choice(n, size=120_000, replace=False)]

    # немного фильтра: убираем явный шум — почти-белые и почти-чёрные забьём позже
    mask_white = np.all(pixels > 248, axis=1)
    mask_black = np.all(pixels < 7, axis=1)
    core = pixels[~(mask_white | mask_black)]
    if core.shape[0] < 500:
        core = pixels  # слишком однотонное — работаем со всем

    # кластеризуем с запасом, потом схлопываем
    k_init = max(need * 3, need + 6)
    centers, labels = kmeans(core, k=k_init, iters=14, seed=SEED)
    # считаем размеры кластеров
    counts = np.bincount(labels, minlength=centers.shape[0])

    # чистим дубли и набираем разнообразие
    picked = ensure_diverse_palette(centers, counts, need)

    # дополнительная сортировка: по покрытию (поиск ближайшего центра)
    # — чтобы сверху шли более "доминирующие" тона
    def nearest_count(c):
        # находим ближайший центр из исходных и берём его размер
        dif = np.sum((centers.astype(np.int16) - np.array(c, np.int16)) ** 2, axis=1)
        j = int(np.argmin(dif))
        return int(counts[j])

    picked.sort(key=nearest_count, reverse=True)
    return picked[:need]


# ── рендер карточки 3×4 ───────────────────────────────────────────────────────
def render_palette_card(colors: List[Tuple[int, int, int]]) -> bytes:
    cols, rows = 3, 4
    assert len(colors) >= cols * rows
    sw, sh = 260, 160            # размер плитки
    pad = 24                      # отступы
    gap = 18                      # расстояние между плитками
    label_h = 42                  # место под HEX

    W = pad * 2 + cols * sw + (cols - 1) * gap
    H = pad * 2 + rows * (sh + label_h) + (rows - 1) * gap
    img = Image.new("RGB", (W, H), (250, 250, 250))
    draw = ImageDraw.Draw(img)

    try:
        # системный шрифт может отсутствовать — не критично
        font = ImageFont.truetype("DejaVuSans.ttf", 26)
    except Exception:
        font = ImageFont.load_default()

    k = 0
    for r in range(rows):
        for c in range(cols):
            x = pad + c * (sw + gap)
            y = pad + r * (sh + label_h + gap)

            color = colors[k]
            k += 1

            # прямоугольник цвета
            draw.rectangle([x, y, x + sw, y + sh], fill=color, outline=(230, 230, 230))

            # подпись HEX
            text = hex_color(color)
            # заменяем устаревший textsize -> textbbox
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            tx = x + (sw - tw) // 2
            ty = y + sh + (label_h - th) // 2

            # легкая подложка для читабельности
            draw.rounded_rectangle([tx - 8, ty - 4, tx + tw + 8, ty + th + 4], radius=6, fill=(255, 255, 255))
            draw.text((tx, ty), text, fill=(30, 30, 30), font=font)

    bio = io.BytesIO()
    img.save(bio, format="PNG", optimize=True)
    return bio.getvalue()


# ── телеграм-бот ──────────────────────────────────────────────────────────────
bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()


@dp.message(Command("start"))
async def on_start(message: Message):
    await message.answer(WELCOME)


@dp.message(F.photo)
async def on_photo(message: Message):
    try:
        # скачиваем лучшее качество
        file = await bot.download(message.photo[-1].file_id)
        file.seek(0)
        img = Image.open(file)

        colors = extract_palette(img, need=NUM_COLORS)
        png_bytes = render_palette_card(colors)

        await message.answer_photo(
            photo=BufferedInputFile(png_bytes, filename="palette.png"),
            caption="Палитра: " + " ".join(hex_color(c) for c in colors[:NUM_COLORS])
        )
    except Exception as e:
        # не падаем и даём понятное сообщение
        await message.answer(f"Ошибка обработки: {type(e).__name__}. Попробуйте другое фото или ещё раз.")


async def main():
    print("color-bot: бот запущен.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
