import os
import io
import logging
from typing import List, Tuple

from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message,
    BufferedInputFile,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from aiogram.filters import CommandStart
from aiogram.enums import ChatMemberStatus

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import MiniBatchKMeans


# =========================
# НАСТРОЙКИ
# =========================
CHANNEL_USERNAME = "desbalances"     # ваш канал @desbalances (без @)
NUM_COLORS = 12                      # сколько цветов отдаём в палитре
MAX_IMAGE_SIDE = 1024                # как сильно уменьшаем фото перед анализом
SWATCHES_PER_ROW = 4                 # сетка 3x4
PADDING = 32                         # отступы в карточке палитры
GAP = 16                             # зазоры между плашками
LABEL_HEIGHT = 36                    # место под подпись HEX
BG_COLOR = (245, 245, 245)           # фон
TEXT_COLOR = (30, 30, 30)            # текст
FONT_PATH = None                     # можно положить .ttf рядом и указать имя файла


# =========================
# ИНИЦИАЛИЗАЦИЯ
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("color-bot")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Переменная окружения TELEGRAM_BOT_TOKEN не задана.")

bot = Bot(token=BOT_TOKEN)  # В aiogram 3: parse_mode сюда НЕ передаём
dp = Dispatcher()


# =========================
# ВСПОМОГАТЕЛЬНОЕ
# =========================
def rgb_to_hex(color: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def resize_image(im: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    w, h = im.size
    scale = max(w, h) / float(max_side)
    if scale > 1.0:
        im = im.resize((int(w / scale), int(h / scale)), Image.LANCZOS)
    return im


def preprocess_pixels(im: Image.Image) -> np.ndarray:
    """
    Готовим пиксели к кластеризации:
    - уменьшаем изображение;
    - переводим в RGB;
    - слегка «разбавляем» выборку по яркости/насыщенности,
      чтобы палитра получалась разнообразнее.
    """
    im = im.convert("RGB")
    im = resize_image(im, MAX_IMAGE_SIDE)
    arr = np.asarray(im, dtype=np.uint8)
    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3)

    # Рассчитываем HSV «по‑бедному», без внешних зависимостей
    # (нам нужны только V и S для стратифицированной подвыборки)
    rgb = pixels.astype(np.float32) / 255.0
    maxc = rgb.max(axis=1)
    minc = rgb.min(axis=1)
    v = maxc
    s = np.where(maxc == 0, 0, (maxc - minc) / maxc)

    # Стратифицированная выборка: берём побольше ярких/насыщенных пикселей,
    # но не забываем и про тихие тона.
    idx = np.arange(pixels.shape[0])
    weight = 0.6 * v + 0.4 * s  # 0..1
    weight = weight + 0.15      # базовый вес, чтобы не занулить серые
    prob = weight / weight.sum()

    sample_size = min(100_000, pixels.shape[0])  # достаточно для устойчивой KMeans
    chosen = np.random.choice(idx, size=sample_size, replace=False, p=prob)
    return pixels[chosen]


def unique_colors(colors: np.ndarray, min_dist: float = 22.0) -> List[Tuple[int, int, int]]:
    """
    Удаляем слишком похожие кластеры.
    min_dist — евклидова дистанция в RGB (0..442). 22 ~ заметная разница.
    """
    sel = []
    for c in colors:
        c = c.astype(int)
        if not sel:
            sel.append(c)
            continue
        d = np.sqrt(((np.array(sel) - c) ** 2).sum(axis=1))
        if (d >= min_dist).all():
            sel.append(c)
    return [tuple(map(int, x)) for x in sel]


def extract_palette(im: Image.Image, n: int = NUM_COLORS) -> List[Tuple[int, int, int]]:
    """
    KMeans по RGB + пост‑обработчик для разнообразия.
    """
    px = preprocess_pixels(im)
    # Берём чуть больше, а потом отфильтруем похожие
    n_clusters = int(n * 1.8)
    n_clusters = max(n_clusters, n + 2)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=4096,
        max_no_improvement=20,
        n_init="auto",
    )
    kmeans.fit(px)
    centers = np.clip(kmeans.cluster_centers_.round(), 0, 255).astype(np.uint8)

    # Сортируем по яркости (V)
    rgb = centers.astype(np.float32) / 255.0
    v = rgb.max(axis=1)
    order = np.argsort(v)[::-1]  # от ярких к тёмным
    centers = centers[order]

    # Убираем дубликаты
    palette = unique_colors(centers, min_dist=22.0)

    # Если недостаточно — добавим из исходных центров
    i = 0
    while len(palette) < n and i < len(centers):
        c = tuple(map(int, centers[i]))
        if c not in palette:
            palette.append(c)
        i += 1

    # Ровно n цветов
    return palette[:n]


def draw_palette_card(
    palette: List[Tuple[int, int, int]],
    swatches_per_row: int = SWATCHES_PER_ROW,
    padding: int = PADDING,
    gap: int = GAP,
    label_h: int = LABEL_HEIGHT,
) -> Image.Image:
    rows = int(np.ceil(len(palette) / swatches_per_row))
    # прикинем размеры карточки
    swatch_w = 220
    swatch_h = 140
    card_w = padding * 2 + swatches_per_row * swatch_w + (swatches_per_row - 1) * gap
    card_h = padding * 2 + rows * (swatch_h + label_h) + (rows - 1) * gap

    img = Image.new("RGB", (card_w, card_h), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Шрифт
    font = None
    if FONT_PATH and os.path.exists(FONT_PATH):
        try:
            font = ImageFont.truetype(FONT_PATH, 22)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    x = padding
    y = padding

    for i, color in enumerate(palette):
        # плашка
        draw.rectangle(
            [x, y, x + swatch_w, y + swatch_h],
            fill=color,
            outline=(220, 220, 220),
            width=2,
        )
        # подпись
        hex_text = rgb_to_hex(color)
        tw, th = draw.textsize(hex_text, font=font)
        tx = x + (swatch_w - tw) // 2
        ty = y + swatch_h + (label_h - th) // 2
        draw.text((tx, ty), hex_text, font=font, fill=TEXT_COLOR)

        # следующий слот
        if (i + 1) % swatches_per_row == 0:
            x = padding
            y += swatch_h + label_h + gap
        else:
            x += swatch_w + gap

    return img


async def is_channel_subscriber(user_id: int) -> bool:
    """
    true — подписан/админ/создатель; false — нет доступа.
    """
    try:
        member = await bot.get_chat_member(chat_id=f"@{CHANNEL_USERNAME}", user_id=user_id)
        return member.status in {
            ChatMemberStatus.ADMINISTRATOR,
            ChatMemberStatus.CREATOR,
            ChatMemberStatus.MEMBER,
        }
    except Exception as e:
        logger.warning("Не удалось проверить подписку: %s", e)
        return False


def subscribe_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(
                text="📌 Подписаться",
                url=f"https://t.me/{CHANNEL_USERNAME}"
            ),
            InlineKeyboardButton(
                text="✅ Проверить подписку",
                callback_data="check_sub"
            ),
        ]]
    )


# =========================
# ХЕНДЛЕРЫ
# =========================
@dp.message(CommandStart())
async def on_start(message: Message):
    user = message.from_user
    if not user:
        return

    ok = await is_channel_subscriber(user.id)
    if not ok:
        await message.answer(
            "Этот бот доступен только для подписчиков канала.\n"
            "Подпишитесь и нажмите «Проверить подписку».",
            reply_markup=subscribe_keyboard(),
        )
        return

    await message.answer(
        "Привет! Я —  генератор цветов от ДИЗ БАЛАНС 🎨 "
        "Отправь мне фото, а я тебе отправлю его цветовую палитру в ответ.\n\n"
        "Пришли изображение одним файлом (не альбомом)."
    )


@dp.callback_query(F.data == "check_sub")
async def on_check_sub(cb):
    ok = await is_channel_subscriber(cb.from_user.id)
    if ok:
        await cb.message.edit_text(
            "Готово! Вы подписаны ✅\nТеперь просто пришлите фото — пришлю палитру из 12 цветов."
        )
    else:
        await cb.answer("Пока подписка не найдена 🤷‍♂️", show_alert=True)


@dp.message(F.photo)
async def on_photo(message: Message):
    user = message.from_user
    if not user:
        return

    # защита: не подписан — не обрабатываем
    if not await is_channel_subscriber(user.id):
        await message.answer(
            "Этот бот доступен только подписчикам канала.\n"
            "Подпишитесь и нажмите «Проверить подписку».",
            reply_markup=subscribe_keyboard(),
        )
        return

    try:
        # берём самое большое превью
        photo_size = message.photo[-1]
        buf = io.BytesIO()
        await bot.download(photo_size, destination=buf)
        buf.seek(0)

        with Image.open(buf) as im:
            palette = extract_palette(im, n=NUM_COLORS)
            card = draw_palette_card(palette)

        out = io.BytesIO()
        card.save(out, format="PNG")
        out.seek(0)

        photo = BufferedInputFile(out.getvalue(), filename="palette.png")
        caption_lines = ["Палитра:"]
        caption_lines.extend(rgb_to_hex(c) for c in palette)
        caption = " ".join(caption_lines)

        await message.answer_photo(photo=photo, caption=caption)

    except Exception as e:
        logger.exception("Ошибка обработки изображения: %s", e)
        await message.answer(
            "Не удалось обработать изображение. Попробуйте другое фото."
        )


@dp.message()
async def on_other(message: Message):
    await message.answer("Пришлите, пожалуйста, фото — я сделаю палитру из 12 цветов.")


# =========================
# main
# =========================
async def main():
    logger.info("Бот запускается. Канал: @%s", CHANNEL_USERNAME)
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
