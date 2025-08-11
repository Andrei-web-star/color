import asyncio
import io
import logging
import os
from typing import List, Tuple

import numpy as np
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ChatMemberStatus
from aiogram.exceptions import TelegramConflictError
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, Message
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans


# ========= Конфигурация =========
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Нет переменной окружения TELEGRAM_BOT_TOKEN")

CHANNEL_USERNAME = "@desbalances"  # проверка подписки по username канала
NUM_COLORS = 12                     # сколько цветов в палитре
CARD_COLUMNS = 3
CARD_ROWS = 4
TILE_W, TILE_H = 340, 240           # размер “плитки” цвета
PADDING = 24                        # отступы вокруг сетки
GAP = 18                            # расстояние между плитками
BG_COLOR = (245, 245, 245)          # фон карточки


# ========= Логирование =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)
log = logging.getLogger("color-bot")


# ========= Утилки =========
def pil_to_bytes(pil: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def downscale(img: Image.Image, max_side: int = 800) -> Image.Image:
    w, h = img.size
    scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
    if scale > 1:
        img = img.resize((int(w / scale), int(h / scale)), Image.Resampling.LANCZOS)
    return img


def extract_palette(img: Image.Image, n_final: int = 12, oversample: int = 28) -> List[Tuple[int, int, int]]:
    """
    1) уменьшаем изображение, берём случайную подвыборку пикселей;
    2) KMeans с k=oversample -> грубые центры;
    3) удаляем очень похожие центры;
    4) farthest-point sampling до n_final для разнообразия.
    """
    img = img.convert("RGB")
    img_small = downscale(img, 600)
    X = np.array(img_small).reshape(-1, 3).astype(np.float32)

    # Сэмплим до 50к пикселей для скорости
    if len(X) > 50_000:
        idx = np.random.choice(len(X), 50_000, replace=False)
        X = X[idx]

    # KMeans oversample
    k = max(n_final + 8, oversample)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    km.fit(X)
    centers = km.cluster_centers_.astype(int)

    # Удаляем дублёры (слишком близкие центры)
    keep = []
    thr = 12.0  # порог близости в RGB
    for c in centers:
        if all(np.linalg.norm(c - np.array(p)) > thr for p in keep):
            keep.append(tuple(c.tolist()))
    centers = np.array(keep, dtype=np.float32)

    # Если после чистки центров меньше, чем надо — просто добираем
    if len(centers) <= n_final:
        chosen = centers
    else:
        # Farthest Point Sampling (максимально разнообразные цвета)
        chosen = []
        # стартуем с самого “среднего” центра по сумме расстояний
        dist_sum = ((centers[None, :, :] - centers[:, None, :]) ** 2).sum(axis=2) ** 0.5
        start_idx = int(np.argmin(dist_sum.sum(axis=1)))
        chosen.append(centers[start_idx])

        remain = np.delete(centers, start_idx, axis=0)
        dmin = np.linalg.norm(remain - chosen[0], axis=1)

        for _ in range(n_final - 1):
            j = int(np.argmax(dmin))
            chosen.append(remain[j])
            remain = np.delete(remain, j, axis=0)
            if len(remain) == 0:
                break
            dmin = np.minimum(dmin, np.linalg.norm(remain - chosen[-1], axis=1))

        chosen = np.array(chosen)

    # Сортируем по “светлоте” для аккуратного вида (формула luma)
    def luma(c):
        r, g, b = c
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    colors = sorted([tuple(map(int, c)) for c in chosen], key=luma, reverse=True)
    # финально — ровно n_final (если вдруг больше/меньше)
    colors = (colors + colors[:n_final])[:n_final]
    return colors


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    """Совместимо с Pillow 10/11: вместо textsize используем textbbox."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def render_palette_card(colors: List[Tuple[int, int, int]]) -> Image.Image:
    """Рисуем карточку 3x4 с подписями HEX под каждым цветом."""
    cols, rows = CARD_COLUMNS, CARD_ROWS
    assert len(colors) == cols * rows

    W = PADDING * 2 + cols * TILE_W + (cols - 1) * GAP
    H = PADDING * 2 + rows * TILE_H + (rows - 1) * GAP
    card = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(card)

    # Шрифт: сначала пытаемся взять DejaVuSans, иначе – системный
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 44)
    except Exception:
        font = ImageFont.load_default()

    for i, color in enumerate(colors):
        cx = i % cols
        cy = i // cols
        x = PADDING + cx * (TILE_W + GAP)
        y = PADDING + cy * (TILE_H + GAP)

        # сам цвет
        draw.rounded_rectangle(
            (x, y, x + TILE_W, y + TILE_H - 70),
            radius=32,
            fill=tuple(color),
            outline=(230, 230, 230),
            width=3,
        )

        # подпись
        hex_text = rgb_to_hex(color)
        tw, th = _text_size(draw, hex_text, font)
        tx = x + (TILE_W - tw) // 2
        ty = y + TILE_H - 60

        # контрастный цвет текста
        r, g, b = color
        text_color = (0, 0, 0) if (0.2126*r + 0.7152*g + 0.0722*b) > 140 else (255, 255, 255)

        # лёгкая подложка для читаемости
        draw.rounded_rectangle((tx - 10, ty - 6, tx + tw + 10, ty + th + 6), radius=10, fill=(255, 255, 255, 200))
        draw.text((tx, ty), hex_text, font=font, fill=text_color)

    return card


# ========= Проверка подписки =========
async def is_subscribed(bot: Bot, user_id: int) -> bool:
    try:
        m = await bot.get_chat_member(chat_id=CHANNEL_USERNAME, user_id=user_id)
        return m.status in {ChatMemberStatus.MEMBER, ChatMemberStatus.CREATOR, ChatMemberStatus.ADMINISTRATOR}
    except Exception as e:
        log.warning("check subscription failed: %s", e)
        # если не смогли проверить — считаем не подписан
        return False


# ========= Хэндлеры =========
async def on_start(message: Message, bot: Bot):
    if not await is_subscribed(bot, message.from_user.id):
        kb = (
            "[Подписаться](https://t.me/desbalances)  •  "
            "[Проверить подписку](/start)"
        )
        await message.answer(
            "Чтобы пользоваться генератором, нужно быть **подписчиком канала** @desbalances.\n\n"
            "Нажми «Подписаться», вернись и снова нажми /start.",
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
        await message.answer(kb, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)
        return

    await message.answer(
        "Привет! Я —  генератор цветов от ДИЗ БАЛАНС 🎨 "
        "Отправь мне **фото**, а я тебе отправлю его **цветовую палитру** в ответ.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def on_photo(message: Message, bot: Bot):
    # проверка подписки перед обработкой
    if not await is_subscribed(bot, message.from_user.id):
        await on_start(message, bot)
        return

    try:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        buf = io.BytesIO()
        await bot.download(file, destination=buf)
        buf.seek(0)

        img = Image.open(buf).convert("RGB")
        colors = extract_palette(img, n_final=NUM_COLORS)
        card = render_palette_card(colors)
        png = pil_to_bytes(card, "PNG")

        caption = "Палитра: " + "  ".join(rgb_to_hex(c) for c in colors)

        await message.answer_photo(
            photo=BufferedInputFile(png, filename="palette.png"),
            caption=caption,
        )

    except Exception as e:
        log.exception("process failed")
        await message.answer(
            f"Ошибка обработки: {type(e).__name__}. Попробуйте другое фото или ещё раз."
        )


# ========= Запуск =========
async def main():
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.message.register(on_start, CommandStart())
    dp.message.register(on_photo, F.photo)

    log.info("color-bot: бот запущен. Канал: %s", CHANNEL_USERNAME)

    # защита от «409 Conflict»: просто один polling; если Render создаст дубликат — Telegram отрежет его
    while True:
        try:
            await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
        except TelegramConflictError:
            # другая копия уже читает updates
            log.error("Conflict: уже запущен другой инстанс. Жду и пробую снова…")
            await asyncio.sleep(5)
        except Exception:
            log.exception("Polling crashed, restart in 3s")
            await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())
