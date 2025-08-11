
import os
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.utils.keyboard import ReplyKeyboardBuilder
import asyncio

# Настройки
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # Токен бота из Render → Environment
CHANNEL_ID = "@DesignAssistant"  # username канала или ID

bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()

# Генерация палитры
def generate_palette(image_path, num_colors=12):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((200, 200))  # ускоряем обработку
    data = np.array(image).reshape(-1, 3)

    # кластеризация цветов (k-means)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(data)
    colors = np.array(kmeans.cluster_centers_, dtype=int)

    # Рисуем палитру
    fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=np.array(color) / 255))
        ax.text(i + 0.5, -0.5, '#%02x%02x%02x' % tuple(color),
                ha='center', va='top', fontsize=10)
    ax.set_xlim(0, num_colors)
    ax.set_ylim(0, 1)
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# Команда /start
@dp.message(Command("start"))
async def start_cmd(message: Message):
    await message.answer(
        "Привет! Я — генератор цветов от ДИЗ БАЛАНС 🎨\n"
        "Отправь мне фото, а я тебе пришлю палитру из 12 цветов."
    )

# Обработка фото
@dp.message(F.photo)
async def handle_photo(message: Message):
    try:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        file_path = file.file_path

        # скачиваем файл
        image_data = await bot.download_file(file_path)
        temp_path = "temp.jpg"
        with open(temp_path, "wb") as f:
            f.write(image_data.read())

        # генерируем палитру
        palette_buf = generate_palette(temp_path)

        # отправляем пользователю
        await message.answer_photo(photo=palette_buf, caption="Вот палитра 🎨")

        # отправляем в канал
        palette_buf.seek(0)
        await bot.send_photo(CHANNEL_ID, photo=palette_buf, caption="Новая палитра 🎨")

    except Exception as e:
        await message.answer("Не удалось обработать изображение. Попробуйте другое фото.")
        print("Ошибка:", e)

# Запуск
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
