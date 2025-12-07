# google_services.py
import os
import io
import logging
import asyncio
from openai import AsyncOpenAI  # Библиотека OpenAI для доступа к DeepSeek
import speech_recognition as sr
from pydub import AudioSegment

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- КОНСТАНТЫ ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

if DEEPSEEK_API_KEY:
    # Инициализируем АСИНХРОННОГО клиента
    client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
else:
    logger.error("DEEPSEEK_API_KEY не найден в переменных окружения!")
    client = None

# --- ФУНКЦИИ АУДИО (Google Speech Recognition - бесплатно, без ключа) ---

def convert_ogg_to_wav(ogg_bytes: bytes) -> io.BytesIO:
    """Конвертирует OGG (Telegram) в WAV"""
    try:
        audio = AudioSegment.from_ogg(io.BytesIO(ogg_bytes))
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception as e:
        logger.error(f"Ошибка конвертации аудио: {e}")
        raise e

def recognize_google_sync(wav_io: io.BytesIO, language="ru-RU") -> str:
    """Синхронное распознавание через Google Web Speech API"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data, language=language)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            return f"Ошибка сервиса распознавания: {e}"

async def transcribe_voice_google(audio_bytes: bytes) -> str:
    """Асинхронная обертка для распознавания голоса"""
    try:
        wav_io = await asyncio.to_thread(convert_ogg_to_wav, audio_bytes)
        text = await asyncio.to_thread(recognize_google_sync, wav_io)
        if not text:
            return "Не удалось разобрать речь (тишина или неразборчиво)."
        return text
    except Exception as e:
        return f"Ошибка при обработке голоса: {e}"

# --- ФУНКЦИИ DEEPSEEK (ТЕКСТ) ---

async def correct_text_with_gemini(raw_text: str) -> str:
    """
    Коррекция текста через DeepSeek.
    Имя функции сохранено для совместимости с bot.py
    """
    if not client:
        return "Ошибка: Не настроен API ключ DeepSeek."

    system_prompt = (
        "Ты — профессиональный редактор и корректор русского языка.\n"
        "Твоя задача:\n"
        "1. Исправить орфографические, пунктуационные и грамматические ошибки.\n"
        "2. Разбить текст на предложения (точки, заглавные буквы).\n"
        "3. Удалить мусорные слова (эээ, типа, ну), если они не несут смысла.\n"
        "4. В ответе вернуть ТОЛЬКО исправленный текст без кавычек и вступлений."
    )

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text},
            ],
            stream=False,
            temperature=0.3 # Низкая температура для точности
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"DeepSeek Error: {e}")
        return f"Ошибка нейросети: {e}"

async def explain_correction_gemini(raw_text: str, corrected_text: str, user_question: str) -> str:
    """Объяснение правок через DeepSeek"""
    if not client:
        return "Ошибка: Не настроен API ключ DeepSeek."

    system_prompt = "Ты — дружелюбный учитель русского языка. Твоя цель — кратко объяснить правило."
    
    user_message = (
        f"Исходный текст: {raw_text}\n"
        f"Исправленный текст: {corrected_text}\n"
        f"Вопрос ученика: {user_question}\n"
    )

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Ошибка при запросе к DeepSeek: {e}"
