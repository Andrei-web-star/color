import logging
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import KMeans
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import CommandStart
import asyncio

# ====== НАСТРОЙКИ ======
API_TOKEN = "ТОКЕН_ТВОЕГО_БОТА"
N_COLORS = 12  # сколько цветов в палитре
# =======================

logging.basicConfig(level=logging.INFO)
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# === Функции для обработки цветов ===
def _rgb_to_lab(rgb_arr):
    rgb = rgb_arr / 255.0
    def inv_gamma(u):
        return np.where(u <= 0.04045, u/12.92, ((u+0.055)/1.055)**2.4)
    r, g, b = inv_gamma(rgb[:,0]), inv_gamma(rgb[:,1]), inv_gamma(rgb[:,2])
    X = r*0.4124564 + g*0.3575761 + b*0.1804375
    Y = r*0.2126729 + g*0.7151522 + b*0.0721750
    Z = r*0.0193339 + g*0.1191920 + b*0.9503041
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = X/Xn, Y/Yn, Z/Zn
    def f(t):
        eps, kappa = 216/24389, 24389/27
        return np.where(t > eps, np.cbrt(t), (kappa*t + 16)/116)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return np.stack([L, a, b], axis=1)

def extract_palette(image_bytes: bytes,
                    n_colors: int = 12,
                    ignore_border: float = 0.05,
                    min_L: float = 20, max_L: float = 95,
                    min_chroma: float = 8):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_small = img.copy()
    img_small.thumbnail((600, 600), Image.LANCZOS)

    w, h = img_small.size
    dx, dy = int(w*ignore_border), int(h*ignore_border)
    img_crop = img_small.crop((dx, dy, w-dx, h-dy))

    arr = np.array(img_crop).reshape(-1, 3)
    if arr.size == 0:
        return ['#000000']*n_colors

    lab = _rgb_to_lab(arr)
    L, a, b = lab[:,0], lab[:,1], lab[:,2]
    chroma = np.sqrt(a*a + b*b)

    mask = (L >= min_L) & (L <= max_L) & (chroma >= min_chroma)
    if mask.sum() < max(1000, n_colors*50):
        mask = (L >= 10) & (L <= 98)

    arr_f = arr[mask]
    if len(arr_f) == 0:
        arr_f = arr

    k = min(n_colors, len(arr_f))
    km = KMeans(n_clusters=k, n_init=8, random_state=42)
    km.fit(arr_f)
    centers = km.cluster_centers_.astype(int)

    centers_lab = _rgb_to_lab(centers)
    order = np.argsort(centers_lab[:,0])
    centers = centers[order]

    hex_colors = [f'#{r:02x}{g:02x}{b:02x}' for r,g,b in centers]
    return hex_colors

def build_palette_image(hex_colors,
                        cols=4,
                        swatch_size=220,
                        margin=20,
                        label_height=60):
    rows = int(np.ceil(len(hex_colors)/cols))
    W = cols*swatch_size + (cols+1)*margin
    H = rows*(swatch_size+label_height) + (rows+1)*margin
    canvas = Image.new("RGB", (W, H), (255,255,255))
    draw = ImageDraw.Draw(canvas)

    font = None
    try:
        font = ImageFont.truetype("Arial.ttf", 26)
    except:
        pass

    for i, hexc in enumerate(hex_colors):
        row, col = divmod(i, cols)
        x0 = margin + col*(swatch_size + margin)
        y0 = margin + row*(swatch_size + label_height + margin)
        color = tuple(int(hexc[j:j+2], 16) for j in (1,3,5))
        draw.rectangle([x0, y0, x0+swatch_size, y0+swatch_size], fill=color)
        text_y = y0 + swatch_size + 8
        draw.text((x0+10, text_y), hexc, fill=(0,0,0), font=font)

    return canvas

# === Обработчики бота ===
@dp.message(CommandStart())
async def start(message: Message):
    await message.answer("Привет! Отправь мне фото, и я сделаю палитру 🎨")

@dp.message(lambda m: m.photo)
async def handle_photo(message: Message):
    try:
        file_id = message.photo[-1].file_id
        file = await bot.get_file(file_id)
        file_bytes = await bot.download_file(file.file_path)
        image_bytes = file_bytes.read()

        # Извлекаем палитру
        hex_list = extract_palette(image_bytes, n_colors=N_COLORS)
        palette_img = build_palette_image(hex_list)

        buf = BytesIO()
        palette_img.save(buf, format='PNG')
        buf.seek(0)

        await message.answer_photo(buf, caption="Палитра:\n" + " ".join(hex_list))

    except Exception as e:
        await message.answer(f"Ошибка обработки: {e}")

# === Запуск ===
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
