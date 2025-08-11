import os
import io
import asyncio
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from aiogram.types import (
    Message, CallbackQuery,
    InlineKeyboardMarkup, InlineKeyboardButton
)
from aiogram.enums import ChatMemberStatus

# ────────────────────────────
# Конфиг
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("telegram_bot_token")
if not BOT_TOKEN:
    raise RuntimeError("Env var TELEGRAM_BOT_TOKEN is empty")

# юзернейм канала без @ — можно переопределить переменной окружения CHANNEL_USERNAME
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "desbalances")

START_TEXT = (
    "Привет! Я —  генератор цветов от ДИЗ БАЛАНС 🎨 "
    "Отправь мне фото, а я тебе отправлю его цветовую палитру в ответ."
)

# ────────────────────────────
# Утилиты

def pil_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def extract_dominant_colors(image: Image.Image, k: int = 12) -> List[Tuple[int, int, int]]:
    img = image.convert("RGB")
    max_side = 400
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    arr = np.asarray(img).reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=k, n_init=4, random_state=42)
    labels = kmeans.fit_predict(arr)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    return [tuple(int(x) for x in centers[i]) for i in order]

def draw_palette(colors: List[Tuple[int, int, int]], cols=3, rows=4) -> Image.Image:
    assert cols * rows == len(colors)

    sw, sh = 280, 220        # размер свотча
    pad, gap = 24, 24
    caption_h = 56

    W = pad*2 + cols*sw + (cols-1)*gap
    H = pad*2 + rows*(sh+caption_h) + (rows-1)*gap

    img = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(img)
    font = pil_font(28)

    for i, rgb in enumerate(colors):
        r, c = divmod(i, cols)
        x0 = pad + c*(sw + gap)
        y0 = pad + r*(sh + caption_h + gap)

        d.rounded_rectangle([x0, y0, x0+sw, y0+sh], radius=16, fill=rgb)

        text = to_hex(rgb)
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        tx = x0 + (sw - tw)//2
        ty = y0 + sh + (caption_h - th)//2

        d.rounded_rectangle([x0+12, y0+sh+8, x0+sw-12, y0+sh+caption_h-8],
                            radius=12, fill=(255, 255, 255))
        d.text((tx, ty), text, fill=(40, 40, 40), font=font)

    return img

async def fetch_input_image(bot: Bot, message: types.Message) -> Image.Image | None:
    try:
        photo_size = max(message.photo, key=lambda p: p.file_size or 0)
        file = await bot.get_file(photo_size.file_id)
        buf = io.BytesIO()
        await bot.download(file, destination=buf)
        buf.seek(0)
        return Image.open(buf)
    except Exception:
        return None

async def build_palette(bot: Bot, message: types.Message) -> tuple[io.BytesIO, List[str]] | None:
    pil_img = await fetch_input_image(bot, message)
    if pil_img is None:
        return None
    try:
        colors = extract_dominant_colors(pil_img, k=12)
        palette = draw_palette(colors, cols=3, rows=4)
        out = io.BytesIO()
        palette.save(out, format="PNG", optimize=True)
        out.seek(0)
        return out, [to_hex(c) for c in colors]
    except Exception:
        return None

# ────────────────────────────
# Подписка

async def is_subscribed(bot: Bot, user_id: int) -> bool:
    try:
        m = await bot.get_chat_member(chat_id=f"@{CHANNEL_USERNAME}", user_id=user_id)
        return m.status in {
            ChatMemberStatus.MEMBER,
            ChatMemberStatus.ADMINISTRATOR,
            ChatMemberStatus.CREATOR,
            ChatMemberStatus.OWNER,
        }
    except Exception:
        return False

def subscribe_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="📌 ПОДПИСАТЬСЯ", url=f"https://t.me/{CHANNEL_USERNAME}"),
        InlineKeyboardButton(text="🔄 Проверить подписку", callback_data="check_sub"),
    ]])

# ────────────────────────────
# Хендлеры

async def cmd_start(message: Message, bot: Bot):
    if not await is_subscribed(bot, message.from_user.id):
        await message.answer(
            "Эта функция доступна только для подписчиков канала.\n"
            "Подпишитесь и нажмите «Проверить подписку».",
            reply_markup=subscribe_kb()
        )
        return
    await message.answer(START_TEXT)

@types.CallbackQuery.filter(F.data == "check_sub")
async def on_check_sub(cb: CallbackQuery, bot: Bot):
    if await is_subscribed(bot, cb.from_user.id):
        await cb.message.answer("Спасибо! Подписка подтверждена. Пришлите фото — пришлю палитру из 12 цветов.")
    else:
        await cb.answer("Ещё нет подписки 🤏", show_alert=True)

async def handle_private_photo(message: Message, bot: Bot):
    if not await is_subscribed(bot, message.from_user.id):
        await message.answer(
            "Доступ только для подписчиков канала.\n"
            "Подпишитесь и нажмите «Проверить подписку».",
            reply_markup=subscribe_kb()
        )
        return

    result = await build_palette(bot, message)
    if not result:
        await message.reply("Не удалось обработать изображение. Попробуйте другое фото.")
        return
    img_bytes, hex_list = result
    caption = "Палитра: " + " ".join(hex_list)
    data = img_bytes.getvalue()
    await message.reply_photo(types.BufferedInputFile(data, "palette.png"), caption=caption)

async def handle_channel_photo(channel_post: Message, bot: Bot):
    # для канала проверка не нужна — бот должен быть админом в самом канале
    result = await build_palette(bot, channel_post)
    if not result:
        await bot.send_message(
            channel_post.chat.id,
            "Не удалось обработать изображение. Попробуйте другое фото.",
            reply_to_message_id=channel_post.message_id
        )
        return
    img_bytes, hex_list = result
    caption = "Палитра: " + " ".join(hex_list)
    data = img_bytes.getvalue()
    await bot.send_photo(
        channel_post.chat.id,
        types.BufferedInputFile(data, "palette.png"),
        caption=caption,
        reply_to_message_id=channel_post.message_id
    )

# ────────────────────────────
# Запуск

async def main():
    bot = Bot(token=BOT_TOKEN)  # без DefaultBotProperties и без parse_mode
    dp = Dispatcher()

    dp.message.register(cmd_start, CommandStart())
    dp.callback_query.register(on_check_sub, F.data == "check_sub")

    dp.message.register(handle_private_photo, F.photo)
    dp.channel_post.register(handle_channel_photo, F.photo)

    print("color-bot | Бот запущен. Ждём апдейты…")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
