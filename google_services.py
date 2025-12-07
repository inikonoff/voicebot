# google_services.py
import os
import io
import logging
import asyncio
from openai import AsyncOpenAI
import speech_recognition as sr
from pydub import AudioSegment

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- КОНСТАНТЫ ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if GROQ_API_KEY:
    # Настройка клиента для Groq
    client = AsyncOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
else:
    logger.error("GROQ_API_KEY не найден в переменных окружения!")
    client = None

# ОБНОВЛЕНИЕ: Используем новейшую модель Llama 3.3 (вместо устаревшей Llama 3)
MODEL_NAME = "llama-3.3-70b-versatile"

# --- ФУНКЦИИ АУДИО (Google Speech - без изменений) ---

def convert_ogg_to_wav(ogg_bytes: bytes) -> io.BytesIO:
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
    try:
        wav_io = await asyncio.to_thread(convert_ogg_to_wav, audio_bytes)
        text = await asyncio.to_thread(recognize_google_sync, wav_io)
        if not text:
            return "Не удалось разобрать речь (тишина или неразборчиво)."
        return text
    except Exception as e:
        return f"Ошибка при обработке голоса: {e}"

# --- ФУНКЦИИ GROQ (ТЕКСТ) ---

async def correct_text_with_gemini(raw_text: str) -> str:
    """
    Коррекция текста через Groq (Llama 3.3).
    """
    if not client:
        return "Ошибка: Не настроен API ключ Groq."

    system_prompt = (
      "Ты — профессиональный редактор русского языка. Задача: преобразовать исходный текст в грамотную, чистую и литературную форму:\n"
        "1. Исправь орфографические, пунктуационные, речевые и грамматические ошибки.\n"
        "2. Удали слова-паразиты (ну, короче, типа, эээ), бессмысленные повторы с сохранением смысла фразы.\n"
        "3. Убери междометия, характерные для устной речи, если они не несут смысловой нагрузки.\n"
        "4. Устрани явную корявость, возникшую из-за неправильных пауз и обрывков фраз.\n"
        "5. Если есть матерные, бранные или грубые выражения, замени их на безобидные литературные аналоги, подходящие по смыслу.\n"
        "Верни ТОЛЬКО готовый исправленный текст без кавычек и вступлений.\n"
    )

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text},
            ],
            stream=False,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq Error: {e}")
        return f"Ошибка нейросети: {e}"

async def explain_correction_gemini(raw_text: str, corrected_text: str, user_question: str) -> str:
    """Объяснение правок через Groq"""
    if not client:
        return "Ошибка: Не настроен API ключ Groq."

    system_prompt = "Ты — учитель русского языка. Кратко объясни правило."
    
    user_message = (
        f"Исходный текст: {raw_text}\n"
        f"Исправленный текст: {corrected_text}\n"
        f"Вопрос ученика: {user_question}\n"
    )

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Ошибка при запросе к Groq: {e}"
