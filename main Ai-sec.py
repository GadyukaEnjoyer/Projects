from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
import openai
from openai import OpenAI
import time
import os
from dotenv import load_dotenv
from typing import TypedDict, Optional

load_dotenv()
openai.api_key = os.getenv('API_KEY')

class TranscriptionAPISettings(BaseSettings):
    tmp_dir: str = 'tmp'
    cors_origins: str = '*'
    cors_allow_credentials: bool = True
    cors_allow_methods: str = '*'
    cors_allow_headers: str = '*'
    whisper_model: str = 'large-v2'
    device: str = 'cuda'
    compute_type: str = 'float16'
    batch_size: int = 16
    language_code: str = 'auto'
    hf_api_key: str = ''
    file_loading_chunk_size_mb: int = 1024
    task_cleanup_delay_min: int = 60
    max_file_size_mb: int = 4096
    max_request_body_size_mb: int = 5000

    class Config:
        env_file = 'env/.env.cuda'
        env_file_encoding = 'utf-8'

settings = TranscriptionAPISettings()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(','),
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods.split(','),
    allow_headers=settings.cors_allow_headers.split(','),
    )


class AudioMessage:
  def __init__(self, path):
    self.path = path
    self.audio = open(self.path, 'rb')

  def get_transcribe(self, api_key):
    client = OpenAI(api_key = api_key)
    transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=self.audio,
    response_format="text"
    )
    return transcription

class Message(TypedDict): 
    audio: AudioMessage
    text: Optional[str]
     
@app.get("/create_thread/")
def create_thread(ass_id):
    
    # Вызываем ассистента
    assistant = openai.beta.assistants.retrieve(ass_id)
    
    # Создаём тред
    thread = openai.beta.threads.create()
    return {"thread_id": thread.id}

def check_status(run_id, thread_id):
    run = openai.beta.threads.runs.retrieve(
        thread_id = thread_id,
        run_id = run_id,
    )
    return(run.status)

# Читаем аудио файл, возвращаем текстовую расшифровку
@app.get("/transcribe")
def transribe(path):
    audio = AudioMessage(path)
    message = Message()
    message["audio"] = audio
    message["text"] = message["audio"].get_transcribe(os.getenv('API_KEY'))
    return(message["text"])
    
@app.get("/get_answer/")
def get_answer(text, thread_id, ass_id):

    # Добавляем запрос пользователя
    message = openai.beta.threads.messages.create(
        thread_id = thread_id,
        role = "user",
        content = text)
    
    # Запускаем бота
    run = openai.beta.threads.runs.create(
    thread_id = thread_id,
    assistant_id = ass_id,) 
    
    # Ждём ответа от бота
    status = check_status(run.id,thread_id)
    while status != "completed":
        if status == "in_progress":
            time.sleep(1)
            status = check_status(run.id,thread_id)
        elif status == "requires_action":
            # Вписать исполнение функции
            return{"answer":"Требуется вызов функции"}
            #status = check_status(run.id,thread_id)  
        elif status == "failed":
            return{"answer":"Ошибка"}
        elif status == "expired":
            return{"answer":"Срок действия диалога истёк"}
        elif status == "cancelled":
            return{"answer":"Диалог отменён"}

    # Получаем ответ
    response = openai.beta.threads.messages.list(thread_id = thread_id)
    if response.data:
        answer = response.data[0].content[0].text.value
    return{"answer":answer}