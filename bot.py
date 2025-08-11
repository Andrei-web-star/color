import os, io, asyncio
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from aiogram.exceptions import TelegramBadRequest

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Env var TELEGRAM_BOT_TOKEN is empty")

CHANNEL_USERNAME = "@assistantdesign"

WELCOME_TEXT = ("Привет! Я —  генератор цветов от ДИЗ БАЛАНС 🎨 "
                "Отправь мне фото, а я тебе отправлю его цветовую палитру в ответ.")

def load_font(size:int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "FreeSans.ttf"):
        try: return ImageFont.truetype(name, size=size)
        except: pass
    return ImageFont.load_default()

def to_hex(rgb:Tuple[int,int,int])->str:
    r,g,b = rgb; return f"#{r:02x}{g:02x}{b:02x}"

def dominant_colors(img:Image.Image, k:int=12)->List[Tuple[int,int,int]]:
    im = img.convert("RGB")
    ms = 512; s = min(1.0, ms/max(im.size))
    if s<1.0: im = im.resize((int(im.width*s), int(im.height*s)), Image.LANCZOS)
    q = im.quantize(colors=k, method=Image.MEDIANCUT).convert("RGB")
    cols = q.getcolors(q.width*q.height) or []
    cols.sort(key=lambda t:t[0], reverse=True)
    res = [tuple(map(int, rgb)) for cnt, rgb in cols[:k]]
    while len(res)<k and res: res.append(res[len(res)%len(res)])
    return res[:k]

def build_card(colors:List[Tuple[int,int,int]])->bytes:
    assert len(colors)==12
    cols, rows = 3, 4
    sw, sh = 280, 220
    gap, pad, label_h = 24, 24, 56
    W = pad*2 + cols*sw + (cols-1)*gap
    H = pad*2 + rows*(sh+label_h) + (rows-1)*gap
    img = Image.new("RGB",(W,H),(245,245,245)); d = ImageDraw.Draw(img); font=load_font(28)
    for i, rgb in enumerate(colors):
        r,c = divmod(i, cols)
        x0 = pad + c*(sw+gap); y0 = pad + r*(sh+label_h+gap)
        d.rounded_rectangle([x0,y0,x0+sw,y0+sh], radius=16, fill=rgb)
        text = to_hex(rgb); l,t,r2,b = d.textbbox((0,0), text, font=font)
        tw,th = r2-l, b-t; tx = x0+(sw-tw)//2; ty = y0+sh+(label_h-th)//2
        d.rounded_rectangle([x0+12,y0+sh+8,x0+sw-12,y0+sh+label_h-8], radius=12, fill=(255,255,255))
        d.text((tx,ty), text, fill=(40,40,40), font=font)
    bio = io.BytesIO(); img.save(bio, "PNG"); bio.seek(0); return bio.getvalue()

async def is_subscriber(bot:Bot, user_id:int)->bool:
    try:
        m = await bot.get_chat_member(CHANNEL_USERNAME, user_id)
        return getattr(m, "status","") in ("creator","administrator","member")
    except TelegramBadRequest:
        return False
    except Exception:
        return False

def subscribe_kb():
    return types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="📌 Подписаться", url=f"https://t.me/{CHANNEL_USERNAME.lstrip('@')}")],
        [types.InlineKeyboardButton(text="🔁 Проверить подписку", callback_data="check_sub")]
    ])

async def cmd_start(m:types.Message, bot:Bot):
    if m.chat.type!="private":
        return
    if await is_subscriber(bot, m.from_user.id):
        await m.answer(WELCOME_TEXT)
    else:
        await m.answer("Этот инструмент доступен только подписчикам канала.\n"
                       "Подпишись и нажми «Проверить подписку».",
                       reply_markup=subscribe_kb())

async def cb_check(call:types.CallbackQuery, bot:Bot):
    if await is_subscriber(bot, call.from_user.id):
        await call.message.edit_text(WELCOME_TEXT)
    else:
        await call.answer("Вы ещё не подписаны 😕", show_alert=True)

async def handle_private_photo(m:types.Message, bot:Bot):
    if m.chat.type!="private":
        return
    if not await is_subscriber(bot, m.from_user.id):
        await m.answer("Доступ только для подписчиков канала.", reply_markup=subscribe_kb()); return
    # качаем фото
    try:
        ph = max(m.photo, key=lambda p: p.file_size or 0)
        f = await bot.get_file(ph.file_id)
        buf = io.BytesIO(); await bot.download(f, destination=buf); buf.seek(0)
        img = Image.open(buf).convert("RGB")
    except Exception:
        await m.reply("Не удалось прочитать изображение. Попробуйте другое фото."); return
    # палитра
    try:
        colors = dominant_colors(img, k=12); png = build_card(colors)
        hex_list = " ".join(to_hex(c) for c in colors)
        await m.reply_photo(types.BufferedInputFile(png, "palette.png"),
                            caption=f"Палитра: {hex_list}")
    except Exception:
        await m.reply("Не удалось обработать изображение. Попробуйте другое фото.")

async def main():
    bot = Bot(token=BOT_TOKEN)   # без parse_mode/DefaultBotProperties
    dp = Dispatcher()
    dp.message.register(cmd_start, CommandStart())
    dp.callback_query.register(cb_check, F.data=="check_sub")
    dp.message.register(handle_private_photo, F.photo)
    print("color-bot | DM-only режим. Ждём апдейты…")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
