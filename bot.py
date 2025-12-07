# bot.py
import os
import io
import logging
import asyncio
from dotenv import load_dotenv
from aiohttp import web  # ВАЖНО для Render

from aiogram import Bot, Dispatcher, types, F, html
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardRemove

# Импортируем функции
from google_services import transcribe_voice_google, correct_text_with_gemini, explain_correction_gemini

load_dotenv()

# --- КОНФИГУРАЦИЯ ---
BOT_TOKEN = os.environ.get("BOT_TOKEN")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not BOT_TOKEN:
    exit("Error: BOT_TOKEN not found!")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# История для контекстных вопросов
user_last_context = {}

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def get_forward_label(message: types.Message) -> str:
    """Формирует заголовок 'от кого', если сообщение переслано"""
    if not message.forward_origin:
        return ""
    
    origin = message.forward_origin
    label = ""

    if origin.type == "user":
        name = origin.sender_user.full_name
        label = f"↩️ от {html.bold(name)}:"
    elif origin.type == "hidden_user":
        name = origin.sender_user_name
        label = f"↩️ от {html.bold(name)}:"
    elif origin.type in ("chat", "channel"):
        title = origin.chat.title if origin.chat and origin.chat.title else "Чата"
        label = f"↩️ из {html.bold(title)}:"
    else:
        label = "↩️ Пересланное сообщение:"
    
    return label + "\n\n"

# --- ФЕЙКОВЫЙ ВЕБ-СЕРВЕР (ДЛЯ UPTIMEROBOT) ---
async def health_check(request):
    """Просто возвращает 200 OK, чтобы Render знал, что мы живы"""
    return web.Response(text="Bot is alive!")

async def start_web_server():
    """Запускает маленький сайт на порту, который выдаст Render"""
    app = web.Application()
    app.router.add_get('/', health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    
    # Render передает порт через переменную окружения PORT
    port = int(os.environ.get("PORT", 8080))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    logger.info(f"Web server started on port {port}")

# --- ХЭНДЛЕРЫ ---

@dp.message(Command("start"))
async def start_handler(message: types.Message):
    text = (
        "👋 <b>Я готов к работе.</b>\n\n"
        "🎤 Отправь голосовое — я превращу его в текст.\n"
        "📝 Отправь текст — я исправлю ошибки.\n"
    )
    await message.answer(text, parse_mode="HTML", reply_markup=ReplyKeyboardRemove())

@dp.message(F.voice)
async def voice_handler(message: types.Message):
    user_id = message.from_user.id
    processing_msg = await message.answer("🎧 Слушаю...")

    try:
        header = get_forward_label(message)
        
        file_info = await bot.get_file(message.voice.file_id)
        voice_buffer = io.BytesIO()
        await bot.download_file(file_info.file_path, voice_buffer)
        voice_bytes = voice_buffer.getvalue()

        # 1. Распознавание
        raw_text = await transcribe_voice_google(voice_bytes)

        if raw_text.startswith("Ошибка") or raw_text.startswith("Не удалось"):
            await processing_msg.edit_text(raw_text)
            return

        # 2. Коррекция
        await processing_msg.edit_text("✍️ Исправляю ошибки...")
        corrected_text = await correct_text_with_gemini(raw_text)
        
        user_last_context[user_id] = {'raw': raw_text, 'corrected': corrected_text}
        
        final_text = header + corrected_text

        await processing_msg.delete()
        
        if len(final_text) > 4096:
            for x in range(0, len(final_text), 4096):
                await message.answer(final_text[x:x+4096], parse_mode="HTML")
        else:
            await message.answer(final_text, parse_mode="HTML")

    except Exception as e:
        logger.error(f"Voice error: {e}")
        await processing_msg.edit_text("❌ Ошибка обработки.")

@dp.message(F.text)
async def text_handler(message: types.Message):
    text = message.text.strip()
    user_id = message.from_user.id
    
    if text.startswith("/"): return

    # Проверка на вопрос "почему?"
    question_triggers = ["почему", "объясни", "зачем", "why"]
    is_question = any(text.lower().startswith(t) for t in question_triggers)

    if is_question and user_id in user_last_context:
        ctx = user_last_context[user_id]
        wait_msg = await message.answer("🤔 Анализирую...")
        explanation = await explain_correction_gemini(ctx['raw'], ctx['corrected'], text)
        await wait_msg.delete()
        await message.answer(explanation, parse_mode="HTML")
        return

    # Обычная коррекция
    processing_msg = await message.answer("✍️ Редактирую...")
    try:
        header = get_forward_label(message)
        corrected_text = await correct_text_with_gemini(text)
        
        user_last_context[user_id] = {'raw': text, 'corrected': corrected_text}
        
        final_text = header + corrected_text
        
        await processing_msg.delete()
        await message.answer(final_text, parse_mode="HTML")

    except Exception as e:
        logger.error(f"Text error: {e}")
        await processing_msg.edit_text("❌ Ошибка.")

# --- ГЛАВНАЯ ФУНКЦИЯ ---
async def main():
    print("Bot starting...")
    
    # 1. Сначала запускаем веб-сервер (чтобы Render поставил галочку "Live")
    await start_web_server()
    
    # 2. Потом запускаем бота
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped")
