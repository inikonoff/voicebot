# google_services.py
import os
import io
import logging
import asyncio
from google import genai
from google.genai import types
import speech_recognition as sr
from pydub import AudioSegment

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- КОНСТАНТЫ ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Инициализация клиента
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    logger.error("GEMINI_API_KEY не найден в переменных окружения!")
    client = None

# Список моделей для перебора (от самой новой к старой)
MODELS_TO_TRY = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-pro"  # Самая старая, но надежная
]

# --- ФУНКЦИИ ---

def convert_ogg_to_wav(ogg_bytes: bytes) -> io.BytesIO:
    """Конвертирует OGG (Telegram) в WAV (Google STT) с помощью FFmpeg/Pydub"""
    try:
        audio = AudioSegment.from_ogg(io.BytesIO(ogg_bytes))
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception as e:
        logger.error(f"Ошибка конвертации аудио (FFmpeg установлен?): {e}")
        raise e

def recognize_google_sync(wav_io: io.BytesIO, language="ru-RU") -> str:
    """Синхронная функция распознавания через Google Web Speech API"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)
        try:
            # Используем публичный API Google (бесплатный)
            text = recognizer.recognize_google(audio_data, language=language)
            return text
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

async def generate_content_safe(prompt: str) -> str:
    """Функция, которая перебирает модели, пока не найдет рабочую"""
    if not client:
        return "Ошибка: API ключ не настроен."

    last_error = ""
    
    # Перебираем модели по очереди
    for model_name in MODELS_TO_TRY:
        try:
            # logger.info(f"Пробуем модель: {model_name}")
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            # Если ошибка 404 или другая, запоминаем и идем к следующей модели
            last_error = str(e)
            continue
    
    # Если ни одна модель не сработала
    logger.error(f"Все модели недоступны. Последняя ошибка: {last_error}")
    return f"Не удалось обработать запрос. Ошибка API: {last_error}"

async def correct_text_with_gemini(raw_text: str) -> str:
    """Коррекция текста"""
    prompt = (
        "Ты — профессиональный редактор. Твоя задача отредактировать текст пользователя.\n"
        "Правила:\n"
        "1. Исправь орфографические, пунктуационные и грамматические ошибки.\n"
        "2. Разбей текст на предложения. Обязательно ставь точки и заглавные буквы.\n"
        "3. Удали слова-паразиты (ну, типа, эээ), если они не несут смысла.\n"
        "4. Верни ТОЛЬКО исправленный текст без кавычек и вступлений.\n\n"
        f"Текст: {raw_text}"
    )
    return await generate_content_safe(prompt)

async def explain_correction_gemini(raw_text: str, corrected_text: str, user_question: str) -> str:
    """Объяснение правок"""
    prompt = (
        "Ты — учитель русского языка. Объясни правку.\n"
        f"Было: {raw_text}\n"
        f"Стало: {corrected_text}\n"
        f"Вопрос: {user_question}\n"
    )
    return await generate_content_safe(prompt)
