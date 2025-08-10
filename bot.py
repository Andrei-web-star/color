import asyncio
import logging
import os
from io import BytesIO
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aiogram import Bot, Dispatcher, F, Router, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart

# ----------------------
# Конфиг
# ----------------------
# Токен берём из переменной окружения на Render
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN".upper())
if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не найден в Environment Variables.")

# Твой канал
CHANNEL_USERNAME = "assistantdesign"      # без @
CHANNEL_ID = -1002608787147               # из getChat

# Сколько цветов в палитре
PALETTE_K = 5

# ----------------------
# Логирование
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("color-bot")

# ----------------------
# Aiogram
# ----------------------
router = Router()


@router.message(CommandStart())
async def on_start(message: types.Message) -> None:
    text = (
        "Привет! Я анализирую изображения и вытаскиваю доминирующие цвета.\n\n"
        f"Добавьте меня админом в канал @{CHANNEL_USERNAME}, публикуйте фото — "
        "я пришлю палитру в ответ к посту."
    )
    await message.answer(text)


# Реакция на фото в ЛС (для удобной проверки)
@router.message(F.photo)
async def on_private_photo(message: types.Message, bot: Bot) -> None:
    if message.chat.type != "private":
        return
    await handle_photo(message, bot)


# Фото в канал-постах
@router.channel_post(F.photo)
async def on_channel_photo(message: types.Message, bot: Bot) -> None:
    # Если это наш канал — отвечаем палитрой на пост
    if message.chat.id == CHANNEL_ID:
        await handle_photo(message, bot)
    else:
        # Молча игнорируем чужие каналы (на всякий)
        log.info(f"Получено фото из другого канала: {message.chat.id}")


# ----------------------
# Палитра
# ----------------------
def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def build_palette_image(colors: List[Tuple[int, int, int]]) -> BytesIO:
    """
    Рисуем горизонтальную палитру + подписи HEX.
    Pillow>=10: вместо textsize используем textbbox.
    """
    width, height = 900, 300
    pad = 20
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Шрифт: возьмём default; если есть DejaVuSans от PIL — ок
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    sw = (width - pad * (len(colors) + 1)) // len(colors)
    sh = height - 100

    x = pad
    for c in colors:
        # блок цвета
        draw.rectangle([x, pad, x + sw, pad + sh], fill=tuple(int(v) for v in c))

        # подпись
        hex_text = rgb_to_hex(tuple(int(v) for v in c))
        bbox = draw.textbbox((0, 0), hex_text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x + (sw - tw) // 2
        ty = pad + sh + (height - (pad + sh) - th) // 2
        draw.text((tx, ty), hex_text, fill=(0, 0, 0), font=font)

        x += sw + pad

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def extract_colors(pil_image: Image.Image, k: int = PALETTE_K) -> List[Tuple[int, int, int]]:
    """
    Вычисляем доминирующие цвета через KMeans.
    """
    # Уменьшим, чтобы ускорить кластеризацию
    img = pil_image.convert("RGB").resize((400, 400))
    data = np.asarray(img).reshape(-1, 3).astype(np.float32)

    # KMeans (scikit-learn)
    # Импорт внутри функции, чтобы ускорить cold start
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    km.fit(data)

    centers = km.cluster_centers_.astype(int).tolist()
    # Отсортируем по "весу" (частоте) — через метки
    labels = km.labels_
    counts = np.bincount(labels, minlength=k)
    ordered = [center for _, center in sorted(zip(counts, centers), reverse=True)]
    return [tuple(map(int, c)) for c in ordered]


async def handle_photo(message: types.Message, bot: Bot) -> None:
    """
    Скачиваем фото, считаем палитру и отправляем изображение-палитру
    ответом на исходный пост (или сообщением в ЛС).
    """
    try:
        # 1) скачиваем самую большую версию фото
        photo = message.photo[-1]
        buf = BytesIO()
        await bot.download(photo, buf)
        buf.seek(0)

        # 2) PIL image
        pil = Image.open(buf).convert("RGB")

        # 3) палитра
        colors = extract_colors(pil, k=PALETTE_K)
        palette_img = build_palette_image(colors)

        caption = "Палитра доминирующих цветов"
        # 4) отвечаем там же
        if message.chat.type == "channel":
            # ответ на пост в канале
            await bot.send_photo(
                chat_id=message.chat.id,
                photo=palette_img,
                caption=caption,
                reply_to_message_id=message.message_id
            )
        else:
            # ЛС
            await message.answer_photo(palette_img, caption=caption)

    except Exception as e:
        log.exception("Ошибка при обработке фото: %s", e)
        try:
            await message.answer("Не удалось обработать изображение. Попробуйте другое фото.")
        except Exception:
            pass


# ----------------------
# entrypoint
# ----------------------
async def main() -> None:
    # Инициализация бота (без устаревших аргументов)
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()
    dp.include_router(router)

    log.info("Бот запущен. Канал: @%s (id=%s)", CHANNEL_USERNAME, CHANNEL_ID)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
