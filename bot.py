import os
import io
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode, ChatMemberStatus
from aiogram.client.default_bot_properties import DefaultBotProperties
from aiogram.filters import CommandStart
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.types.input_file import BufferedInputFile

from sklearn.cluster import KMeans

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME")  # например: "assistantdesign" или "desbalances"

NUM_COLORS = 12
PREVIEW_MAX = 640  # до такого размера уменьшаем длинную сторону перед анализом


# ---------- утилиты ----------

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def extract_palette(img: Image.Image, n_colors: int = NUM_COLORS) -> list[tuple[int, int, int]]:
    # уменьшение для ускорения
    w, h = img.size
    scale = PREVIEW_MAX / max(w, h) if max(w, h) > PREVIEW_MAX else 1.0
    if scale != 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    flat = arr.reshape(-1, 3)

    # Лёгкая фильтрация: прибираем почти белые и почти чёрные пиксели (блики/шумы)
    brightness = flat.mean(axis=1)
    mask = (brightness > 8) & (brightness < 247)
    flat = flat[mask]
    if flat.shape[0] < n_colors * 20:
        # слишком мало данных после фильтра — откатываем фильтр
        flat = arr.reshape(-1, 3)

    # KMeans
    kmeans = KMeans(n_clusters=n_colors, n_init=6, random_state=42)
    labels = kmeans.fit_predict(flat)
    centers = np.clip(np.rint(kmeans.cluster_centers_), 0, 255).astype(np.uint8)

    # сортировка по размеру кластера (частота)
    counts = Counter(labels)
    order = [idx for idx, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
    # индексы центров в порядке убывания частоты
    # но у нас order ссылается на метки, нужно сопоставить к центрам:
    # создадим mapping: метка -> центр
    label_to_center = {label: centers[label] for label in range(len(centers))}
    ordered = [tuple(int(c) for c in label_to_center[l]) for l in order]

    # иногда kmeans даёт очень близкие цвета — немного разрежим
    dedup = []
    for c in ordered:
        if all(np.linalg.norm(np.array(c) - np.array(p)) >= 10 for p in dedup):
            dedup.append(c)
        if len(dedup) == n_colors:
            break
    # если после разрежения не хватило цветов — добьём исходными
    i = 0
    while len(dedup) < n_colors and i < len(ordered):
        if ordered[i] not in dedup:
            dedup.append(ordered[i])
        i += 1

    return dedup[:n_colors]


def render_palette_card(colors: list[tuple[int, int, int]]) -> bytes:
    # сетка 3×4
    cols, rows = 3, 4
    sw = 320     # ширина плашки
    sh = 220     # высота плашки
    pad = 24     # внутренний паддинг между плашками
    outer = 30   # внешние поля

    card_w = outer * 2 + cols * sw + (cols - 1) * pad
    card_h = outer * 2 + rows * sh + (rows - 1) * pad
    card = Image.new("RGB", (card_w, card_h), (245, 245, 245))
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default()

    for i, rgb in enumerate(colors[: rows * cols]):
        r, c = divmod(i, cols)
        x0 = outer + c * (sw + pad)
        y0 = outer + r * (sh + pad)
        swatch = Image.new("RGB", (sw, sh), rgb)
        card.paste(swatch, (x0, y0))

        # подпись HEX на светлой/тёмной подложке
        hex_ = rgb_to_hex(rgb)
        text = hex_
        # оценим контраст (простая яркость)
        bright = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
        text_color = (0, 0, 0) if bright > 160 else (255, 255, 255)

        # позиционирование текста — через textbbox (в Pillow 10 нет textsize)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x0 + sw - tw - 10
        ty = y0 + sh - th - 8

        # лёгкая тень для читаемости
        shadow = (0, 0, 0) if text_color == (255, 255, 255) else (255, 255, 255)
        draw.text((tx + 1, ty + 1), text, font=font, fill=shadow)
        draw.text((tx, ty), text, font=font, fill=text_color)

    buf = io.BytesIO()
    card.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# ---------- aiogram ----------

dp = Dispatcher()

@dp.message(CommandStart())
async def cmd_start(message: Message, bot: Bot):
    if CHANNEL_USERNAME:
        try:
            member = await bot.get_chat_member(chat_id=f"@{CHANNEL_USERNAME}", user_id=message.from_user.id)
            if member.status not in (
                ChatMemberStatus.MEMBER, ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR, ChatMemberStatus.OWNER
            ):
                raise Exception("not_subscribed")
        except Exception:
            kb = InlineKeyboardMarkup(inline_keyboard=[[
                InlineKeyboardButton(text="📌 Подписаться", url=f"https://t.me/{CHANNEL_USERNAME}"),
                InlineKeyboardButton(text="🔄 Проверить подписку", callback_data="check_sub"),
            ]])
            await message.answer(
                "Эта функция доступна только подписчикам канала.\nПодпишитесь и нажмите «Проверить подписку».",
                reply_markup=kb
            )
            return

    await message.answer(
        "Привет! Я —  генератор цветов от ДИЗ БАЛАНС 🎨 "
        "Отправь мне фото, а я тебе отправлю его цветовую палитру в ответ."
    )


@dp.callback_query(F.data == "check_sub")
async def on_check_subscription(cb, bot: Bot):
    try:
        member = await bot.get_chat_member(chat_id=f"@{CHANNEL_USERNAME}", user_id=cb.from_user.id)
        ok = member.status in (
            ChatMemberStatus.MEMBER, ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR, ChatMemberStatus.OWNER
        )
    except Exception:
        ok = False

    if ok:
        await cb.message.answer(
            "Спасибо! Подписка подтверждена. Пришлите фото — пришлю палитру из 12 цветов."
        )
    else:
        await cb.answer("Ещё нет подписки 🤏", show_alert=True)


@dp.message(F.photo)
async def on_photo(message: Message, bot: Bot):
    try:
        # скачиваем самое большое превью
        photo = message.photo[-1]
        buf = io.BytesIO()
        await bot.download(photo, destination=buf)
        buf.seek(0)

        img = Image.open(buf).convert("RGB")
        colors = extract_palette(img, n_colors=NUM_COLORS)

        png_bytes = render_palette_card(colors)
        hex_line = " ".join(rgb_to_hex(c) for c in colors)

        await message.answer_photo(
            BufferedInputFile(png_bytes, filename="palette.png"),
            caption=f"Палитра: {hex_line}"
        )
    except Exception as e:
        await message.answer(
            f"Ошибка обработки: {type(e).__name__}. Попробуйте другое фото или ещё раз."
        )


async def main():
    # Bot c правильной инициализацией для aiogram 3.7+
    bot = Bot(
        token=BOT_TOKEN,
        default_bot_properties=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    print(f"color-bot: бот запущен. Канал: @{CHANNEL_USERNAME}" if CHANNEL_USERNAME else "color-bot: бот запущен.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
