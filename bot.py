# bot.py
import os
import io
import logging
import asyncio
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardRemove

# Импортируем наши новые Google-функции
from google_services import transcribe_voice_google, correct_text_with_gemini, explain_correction_gemini

# Загружаем переменные окружения (.env)
load_dotenv()

# --- КОНФИГУРАЦИЯ ---
BOT_TOKEN = os.environ.get("BOT_TOKEN")

# Включаем логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not BOT_TOKEN:
    exit("Error: BOT_TOKEN not found in environment variables!")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Хранилище истории для контекстных вопросов (в памяти)
# Формат: {user_id: {'raw': str, 'corrected': str}}
user_last_context = {}

# --- ХЭНДЛЕРЫ ---

@dp.message(Command("start"))
async def start_handler(message: types.Message):
    text = (
        "👋 <b>Привет! Я бесплатный ИИ-редактор.</b>\n\n"
        "🎤 <b>Голосовые:</b> Пришли мне голосовое любой длины — я переведу его в текст и расставлю знаки препинания.\n"
        "📝 <b>Текст:</b> Напиши или перешли мне любой черновик — я исправлю ошибки.\n\n"
        "<i>Работаю на базе Google Speech API и Gemini.</i>"
    )
    await message.answer(text, parse_mode="HTML", reply_markup=ReplyKeyboardRemove())

@dp.message(F.voice)
async def voice_handler(message: types.Message):
    """Обработка голосовых сообщений"""
    user_id = message.from_user.id
    processing_msg = await message.answer("🎧 Слушаю и обрабатываю...")

    try:
        # --- БЛОК ОПРЕДЕЛЕНИЯ АВТОРА (для пересланных) ---
        author_info = ""
        if message.forward_origin:
            # В aiogram 3.x информация о пересылке лежит в forward_origin
            origin = message.forward_origin
            
            sender_name = "Неизвестный"
            
            if origin.type == "user":
                sender_name = origin.sender_user.full_name
            elif origin.type == "chat":
                sender_name = origin.chat.title
            elif origin.type == "channel":
                sender_name = origin.chat.title
            elif origin.type == "hidden_user":
                sender_name = origin.sender_user_name
            
            author_info = f"🗣 <b>От: {sender_name}</b>\n\n"
        # -----------------------------------------------------

        # Скачиваем файл
        file_info = await bot.get_file(message.voice.file_id)
        voice_buffer = io.BytesIO()
        await bot.download_file(file_info.file_path, voice_buffer)
        voice_bytes = voice_buffer.getvalue()

        # 1. Распознавание
        raw_text = await transcribe_voice_google(voice_bytes)

        if raw_text.startswith("Ошибка") or raw_text.startswith("Не удалось"):
            await processing_msg.edit_text(raw_text)
            return

        # 2. Редактура
        await processing_msg.edit_text("✍️ Текст распознан. Исправляю ошибки...")
        corrected_text = await correct_text_with_gemini(raw_text)

        # Сохраняем контекст
        user_last_context[user_id] = {'raw': raw_text, 'corrected': corrected_text}

        # Формируем итоговый текст с автором (если есть)
        final_text = author_info + corrected_text

        await processing_msg.delete()
        
        # Отправка (с учетом лимита телеграм)
        if len(final_text) > 4096:
            for x in range(0, len(final_text), 4096):
                await message.answer(final_text[x:x+4096], parse_mode="HTML")
        else:
            await message.answer(final_text, parse_mode="HTML")

    except Exception as e:
        logger.error(f"Critical error: {e}")
        await processing_msg.edit_text("❌ Произошла ошибка при обработке.")
@dp.message(F.text)
async def text_handler(message: types.Message):
    """Обработка текстовых сообщений"""
    text = message.text.strip()
    user_id = message.from_user.id
    
    # Игнорируем команды
    if text.startswith("/"):
        return

    # Проверка на вопрос "почему?"
    question_triggers = ["почему", "объясни", "зачем"]
    is_question = any(text.lower().startswith(t) for t in question_triggers)

    if is_question and user_id in user_last_context:
        # Это вопрос к предыдущему исправлению
        ctx = user_last_context[user_id]
        wait_msg = await message.answer("🤔 Анализирую правки...")
        explanation = await explain_correction_gemini(ctx['raw'], ctx['corrected'], text)
        await wait_msg.delete()
        await message.answer(explanation)
        return

    # Обычная коррекция текста
    processing_msg = await message.answer("✍️ Читаю и редактирую...")
    
    try:
        corrected_text = await correct_text_with_gemini(text)
        
        user_last_context[user_id] = {'raw': text, 'corrected': corrected_text}
        
        await processing_msg.delete()
        await message.answer(corrected_text)
        # Опционально удаляем исходное сообщение пользователя для чистоты
        # try: await message.delete() 
        # except: pass

    except Exception as e:
        logger.error(f"Text error: {e}")
        await processing_msg.edit_text("❌ Ошибка обработки текста.")

# --- ЗАПУСК ---
async def main():
    print("Bot started...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped")
