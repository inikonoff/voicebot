# google_services.py
import os
import io
import logging
import asyncio
from google import genai
from google.genai import types # Типы для конфигурации
import speech_recognition as sr
from pydub import AudioSegment

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- КОНСТАНТЫ ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    logger.error("GEMINI_API_KEY не найден в переменных окружения!")
    client = None

# Список моделей для перебора.
# gemini-2.0-flash стоит первой, так как она новее и быстрее.
MODELS_TO_TRY = [
    "gemini-2.0-flash",      # Новинка
    "gemini-2.0-flash-lite", # Облегченная версия 2.0
]

# --- ФУНКЦИИ АУДИО ---

def convert_ogg_to_wav(ogg_bytes: bytes) -> io.BytesIO:
    """Конвертирует OGG (Telegram) в WAV (Google STT)"""
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

# --- ФУНКЦИИ GEMINI (ТЕКСТ) ---

async def generate_content_safe(user_text: str, system_instruction: str) -> str:
    """
    Универсальная функция запроса к Gemini.
    Перебирает модели и использует system_instruction (как в доке).
    """
    if not client:
        return "Ошибка: API ключ не настроен."

    last_error = ""
    
    # Конфигурация с системной инструкцией
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.3 # Меньше креатива, больше точности
    )

    # Перебираем модели по очереди
    for model_name in MODELS_TO_TRY:
        try:
            # logger.info(f"Пробуем модель: {model_name}")
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=user_text,
                config=config
            )
            return response.text.strip()
        except Exception as e:
            # Если 404 (модель не найдена) или другая ошибка - пробуем следующую
            last_error = str(e)
            continue
    
    logger.error(f"Все модели недоступны. Ошибка: {last_error}")
    return f"Не удалось обработать запрос. Проверьте API Key. Ошибка: {last_error}"

async def correct_text_with_gemini(raw_text: str) -> str:
    """Коррекция текста"""
    
    # Системная инструкция (System Prompt)
    sys_inst = (
        "Ты — профессиональный редактор и корректор.\n"
        "Твоя задача:\n"
        "1. Исправить орфографические, пунктуационные и грамматические ошибки.\n"
        "2. Разбить текст на предложения (точки, заглавные буквы).\n"
        "3. Удалить мусорные слова (эээ, типа, ну), если они не несут смысла.\n"
        "4. В ответе вернуть ТОЛЬКО исправленный текст."
    )
    
    return await generate_content_safe(user_text=raw_text, system_instruction=sys_inst)

async def explain_correction_gemini(raw_text: str, corrected_text: str, user_question: str) -> str:
    """Объяснение правок"""
    
    # Системная инструкция
    sys_inst = "Ты — дружелюбный учитель русского языка. Твоя цель — кратко объяснить правило."
    
    # Текст пользователя (Prompt)
    user_message = (
        f"Исходный текст: {raw_text}\n"
        f"Исправленный текст: {corrected_text}\n"
        f"Вопрос ученика: {user_question}\n"
    )
    
    return await generate_content_safe(user_text=user_message, system_instruction=sys_inst)
