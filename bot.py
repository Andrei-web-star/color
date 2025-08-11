import logging
import os
import io
from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.types import FSInputFile
from aiogram.utils.markdown import hlink
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Токен бота из переменных окружения
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Создание бота и диспетчера
bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()

# Храним последнюю палитру для команды /pdf
last_palette_image = None
last_palette_colors = None

# Приветствие (как было раньше)
@dp.message(commands=["start"])
async def start(message: types.Message):
    await message.answer(
        "Привет! Я — анализатор изображений и вытаскиваю доминирующие цвета.\n\n"
        "Добавьте меня админом в канал @assistantdesign, публикуйте фото — я пришлю палитру в ответ к посту."
    )

# Функция для извлечения доминирующих цветов
def extract_colors(image_path, num_colors=12):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((200, 200))  # Уменьшаем размер для ускорения обработки
    np_image = np.array(image)
    np_image = np_image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(np_image)
    colors = kmeans.cluster_centers_.astype(int)

    hex_colors = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in colors]
    return hex_colors

# Функция для генерации картинки палитры
def create_palette_image(colors):
    global last_palette_image
    fig, ax = plt.subplots(1, len(colors), figsize=(len(colors) * 2, 2))
    if len(colors) == 1:
        ax = [ax]
    for i, color in enumerate(colors):
        ax[i].imshow(np.ones((10, 10, 3), dtype=np.uint8) * np.array(Image.new("RGB", (1, 1), color).getpixel((0, 0))))
        ax[i].axis("off")
        ax[i].set_title(color, fontsize=8)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    last_palette_image = buf
    plt.close(fig)
    return buf

# Обработка фото
@dp.message(content_types=["photo"])
async def handle_photo(message: types.Message):
    global last_palette_colors
    try:
        photo = message.photo[-1]
        file_path = await bot.download(photo)
        colors = extract_colors(file_path, num_colors=12)
        last_palette_colors = colors
        palette_img = create_palette_image(colors)

        await message.reply_photo(palette_img, caption="Палитра: " + " ".join([hlink(c, f"https://www.color-hex.com/color/{c[1:]}") for c in colors]))
    except Exception as e:
        await message.reply("Не удалось обработать изображение. Попробуйте другое фото.")
        logging.error(f"Ошибка обработки фото: {e}")

# Сохранение в PDF
@dp.message(commands=["pdf"])
async def save_pdf(message: types.Message):
    global last_palette_image, last_palette_colors
    if last_palette_image is None or last_palette_colors is None:
        await message.reply("Сначала отправьте фото, чтобы я создал палитру.")
        return

    pdf_path = "palette.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 750, "Цветовая палитра")
    y = 700
    for color in last_palette_colors:
        c.setFillColorRGB(int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:], 16)/255)
        c.rect(100, y, 50, 20, fill=True, stroke=False)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(160, y + 5, color)
        y -= 30
    c.save()

    await message.reply_document(FSInputFile(pdf_path))

# Запуск бота
if __name__ == "__main__":
    import asyncio
    asyncio.run(dp.start_polling(bot))
