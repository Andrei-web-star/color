import os
import io
import math
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message, BufferedInputFile, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.enums import ChatType

# ================== Настройки ==================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN".upper()) or os.getenv("TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Env var TELEGRAM_BOT_TOKEN is empty")

# канал, подписка на который обязательна
CHANNEL_USERNAME = "desbalances"  # без @
CHANNEL_LINK = f"https://t.me/{CHANNEL_USERNAME}"

# Сколько цветов в палитре
PALETTE_K = 12

# Размер карточки (на глаз под Telegram)
CARD_W, CARD_H = 1024, 1280
GRID_COLS, GRID_ROWS = 3, 4  # 12 ячеек
MARGIN = 32
CELL_GAP = 16
LABEL_H = 54
BORDER = 2

WELCOME_TEXT = (
    "Привет! Я —  генератор цветов от ДИЗ БАЛАНС 🎨 "
    "Отправь мне фото, а я тебе отправлю его цветовую палитру в ответ."
)

# ================== Вспомогательные функции (цвет) ==================
def _to_lab(img: Image.Image) -> np.ndarray:
    lab = img.convert("LAB")
    return np.asarray(lab, dtype=np.float32).reshape(-1, 3)

def _tile_sample(img: Image.Image, per_tile: int = 1200, tiles: int = 6) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    scale = 768 / max(w, h) if max(w, h) > 768 else 1.0
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    W, H = img.size
    lab = np.asarray(img.convert("LAB"), dtype=np.float32)

    xs = np.linspace(0, W, tiles + 1, dtype=int)
    ys = np.linspace(0, H, tiles + 1, dtype=int)
    parts = []
    rng = np.random.default_rng(42)
    for i in range(tiles):
        for j in range(tiles):
            x0, x1 = xs[i], xs[i+1]
            y0, y1 = ys[j], ys[j+1]
            tile = lab[y0:y1, x0:x1, :].reshape(-1, 3)
            if len(tile) == 0:
                continue
            take = min(per_tile, len(tile))
            idx = rng.choice(len(tile), take, replace=False)
            parts.append(tile[idx])
    if not parts:
        return lab.reshape(-1, 3)
    all_lab = np.concatenate(parts, axis=0)

    L = all_lab[:, 0]
    keep = (L > 2) & (L < 98)
    return all_lab[keep]

def _prepare_pixels(im: Image.Image, sample: int = 200_000) -> np.ndarray:
    lab = _tile_sample(im, per_tile=1200, tiles=6)
    if len(lab) > sample:
        idx = np.random.choice(len(lab), sample, replace=False)
        lab = lab[idx]
    return lab  # LAB

def _deltaE76(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def _lab_to_rgb_tuple(lab_vec: np.ndarray) -> Tuple[int, int, int]:
    arr = lab_vec.reshape(1, 1, 3).astype(np.uint8)
    pil_lab = Image.fromarray(arr, mode="LAB").convert("RGB")
    r, g, b = pil_lab.getpixel((0, 0))
    return int(r), int(g), int(b)

def _kmeans_colors(pixels_lab: np.ndarray, k: int) -> List[Tuple[int, int, int]]:
    k_over = max(18, int(k * 2 + 4))
    km = KMeans(n_clusters=k_over, n_init=10, random_state=42)
    labels = km.fit_predict(pixels_lab)
    centers = km.cluster_centers_  # LAB
    counts = np.bincount(labels, minlength=k_over)

    Lvals = centers[:, 0]
    order = np.lexsort((Lvals, -counts))
    centers = centers[order]
    counts = counts[order]

    mean_lab = np.median(pixels_lab, axis=0)
    dists = np.linalg.norm(centers - mean_lab, axis=1)
    pick0 = int(np.argmin(dists))

    selected = [centers[pick0]]

    def binL(L):
        if L < 35: return "dark"
        if L > 70: return "light"
        return "mid"

    quotas = {"dark": 4, "mid": 4, "light": 4}
    quotas[binL(centers[pick0, 0])] -= 1

    min_de = 12.0
    i = 0
    while len(selected) < k and i < len(centers):
        c = centers[i]
        b = binL(c[0])
        if quotas[b] <= 0:
            i += 1
            continue
        ok = True
        for s in selected:
            if _deltaE76(c, s) < min_de:
                ok = False
                break
        if ok:
            selected.append(c)
            quotas[b] -= 1
        i += 1

    i = 0
    while len(selected) < k and i < len(centers):
        c = centers[i]
        ok = True
        for s in selected:
            if _deltaE76(c, s) < (min_de - 3):
                ok = False
                break
        if ok:
            selected.append(c)
        i += 1

    i = 0
    while len(selected) < k and i < len(centers):
        selected.append(centers[i])
        i += 1

    result_rgb = [_lab_to_rgb_tuple(lab) for lab in selected[:k]]

    selected_lab = np.vstack(selected[:k])
    sort_idx = np.lexsort((selected_lab[:, 2], selected_lab[:, 1], selected_lab[:, 0]))
    result_rgb = [result_rgb[i] for i in sort_idx]
    return result_rgb

def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)

# ================== Карточка палитры ==================
def render_palette_card(colors: List[Tuple[int,int,int]], source_preview: Image.Image | None = None) -> Image.Image:
    card = Image.new("RGB", (CARD_W, CARD_H), (245, 246, 250))
    draw = ImageDraw.Draw(card)

    # попытка взять системный шрифт (если нет — PIL default)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 40)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 34)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # заголовок (опционально можно вернуть)
    # draw.text((MARGIN, MARGIN), "Палитра", fill=(30,30,30), font=font)

    grid_top = MARGIN
    grid_left = MARGIN
    grid_right = CARD_W - MARGIN
    grid_bottom = CARD_H - MARGIN

    cell_w = (grid_right - grid_left - (GRID_COLS - 1) * CELL_GAP) // GRID_COLS
    cell_h = (grid_bottom - grid_top - (GRID_ROWS - 1) * CELL_GAP) // GRID_ROWS

    # рисуем 12 ячеек
    idx = 0
    hexes = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if idx >= len(colors):
                break
            x = grid_left + c * (cell_w + CELL_GAP)
            y = grid_top + r * (cell_h + CELL_GAP)

            # прямоугольник цвета (оставим снизу место под подпись)
            rect_h = cell_h
            color = colors[idx]
            draw.rounded_rectangle(
                [x, y, x + cell_w, y + rect_h],
                radius=18,
                fill=color,
                outline=(230, 232, 236),
                width=BORDER
            )

            # подпись HEX (на нижней кромке, в «плашке» полупрозрачной)
            hex_code = _rgb_to_hex(color)
            hexes.append(hex_code)
            text_w, text_h = draw.textbbox((0,0), hex_code, font=font_small)[2:]
            pad = 10
            box_h = text_h + pad*2
            box_y = y + rect_h - box_h
            draw.rectangle([x, box_y, x + cell_w, y + rect_h], fill=(255,255,255,128))
            draw.text((x + (cell_w - text_w)//2, box_y + pad), hex_code, fill=(30,30,30), font=font_small)

            idx += 1

    # маленький превью исходника в левом верхнем углу (необязательно)
    if source_preview is not None:
        prev = source_preview.copy().convert("RGB")
        pw = CARD_W // 5
        ph = int(prev.height / prev.width * pw)
        prev = prev.resize((pw, ph), Image.LANCZOS)
        card.paste(prev, (CARD_W - pw - MARGIN, MARGIN))

    return card, hexes

# ================== Проверка подписки ==================
async def is_subscriber(bot: Bot, user_id: int) -> bool:
    try:
        member = await bot.get_chat_member(chat_id=f"@{CHANNEL_USERNAME}", user_id=user_id)
        status = getattr(member, "status", None)
        # допустимые статусы
        return status in ("creator", "administrator", "member") or getattr(member, "is_member", False)
    except Exception:
        # если канал закрыт для бота или бот не админ в канале — считаем не подписан
        return False

def subscribe_keyboard() -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="📌 Подписаться", url=CHANNEL_LINK),
        InlineKeyboardButton(text="🔄 Проверить подписку", callback_data="check_sub")
    ]])
    return kb

# ================== Aiogram ==================
bot = Bot(token=BOT_TOKEN)  # без parse_mode тут — 3.7.0 это не поддерживает
dp = Dispatcher()

@dp.message(CommandStart())
async def on_start(message: Message):
    # если пришли из поста/кнопки — сразу проверим подписку
    if message.chat.type == ChatType.PRIVATE:
        if await is_subscriber(bot, message.from_user.id):
            await message.answer(WELCOME_TEXT)
        else:
            await message.answer(
                "Этот инструмент доступен только подписчикам канала.\n\n"
                "Подпишитесь и нажмите <b>«Проверить подписку»</b>.",
                parse_mode="HTML",
                reply_markup=subscribe_keyboard()
            )

@dp.callback_query(F.data == "check_sub")
async def on_check_sub(callback):
    uid = callback.from_user.id
    if await is_subscriber(bot, uid):
        await callback.message.edit_text(WELCOME_TEXT)
    else:
        await callback.answer("Пока не вижу подписку. Подпишись и нажми ещё раз.", show_alert=True)

@dp.message(F.photo)
async def on_photo(message: Message):
    # ограничиваем только приватные чаты (боту кидают фото)
    if message.chat.type != ChatType.PRIVATE:
        return

    if not await is_subscriber(bot, message.from_user.id):
        await message.answer(
            "Инструмент только для подписчиков канала.\nПодпишитесь и нажмите «Проверить подписку».",
            reply_markup=subscribe_keyboard()
        )
        return

    try:
        file = await bot.get_file(message.photo[-1].file_id)
        file_bytes = await bot.download_file(file.file_path)
        im = Image.open(io.BytesIO(file_bytes.read())).convert("RGB")

        # выборка пикселей в LAB + «умная» 12‑ка
        pixels_lab = _prepare_pixels(im)
        colors = _kmeans_colors(pixels_lab, PALETTE_K)

        # карточка + список HEX
        preview_for_card = im.copy()
        preview_for_card.thumbnail((480, 480), Image.LANCZOS)
        card_img, hexes = render_palette_card(colors, source_preview=preview_for_card)

        buf = io.BytesIO()
        card_img.save(buf, format="PNG")
        buf.seek(0)

        caption = "Палитра: " + " ".join(hexes)
        await message.answer_photo(
            BufferedInputFile(buf.read(), filename="palette.png"),
            caption=caption,
            parse_mode="HTML"
        )

    except Exception as e:
        await message.answer("Не удалось обработать изображение. Попробуйте другое фото.")
        # можно логнуть, если нужно:
        # print("Error:", e)

# ================== Запуск ==================
if __name__ == "__main__":
    import asyncio
    print("color-bot | Бот запускаем. Канал:", f"@{CHANNEL_USERNAME}")
    asyncio.run(dp.start_polling(bot))
