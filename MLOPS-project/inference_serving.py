from __future__ import annotations

import pickle
import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

from clearml import Model

# --- Вставить сюда model_id после register_model.py ---
REGISTRY_MODEL_ID = "abb6fb0b8a674ebc9f39068db0c884ad"

# Путь для временного хранения модели
MODEL_CACHE_PATH = Path("model_cache")
MODEL_CACHE_PATH.mkdir(exist_ok=True)


class SentimentClassifier:
    """Класс-обертка для модели"""
    
    def __init__(self):
        self.classifier = None
        self.vectorizer = None
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели из ClearML Registry"""
        print("Загрузка модели из ClearML Registry...")
        try:
            weights_path = Model(model_id=REGISTRY_MODEL_ID).get_local_copy()
            with open(weights_path, "rb") as handle:
                payload = pickle.load(handle)
            self.classifier = payload["model"]
            self.vectorizer = payload["vectorizer"]
            print("Модель успешно загружена!")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            raise
    
    def predict(self, text: str) -> dict:
        """Метод для инференса"""
        text = text.strip()
        
        if not text:
            return {"error": "Text field is empty"}
        
        if self.classifier is None or self.vectorizer is None:
            return {"error": "Model not loaded"}
        
        try:
            vectorized = self.vectorizer.transform([text])
            label = self.classifier.predict(vectorized)[0]
            confidence = float(self.classifier.predict_proba(vectorized).max())
            
            return {
                "label": label,
                "confidence": round(confidence, 3)
            }
        except Exception as e:
            return {"error": str(e)}


class RequestHandler(BaseHTTPRequestHandler):
    """Обработчик HTTP запросов"""
    
    classifier = None
    
    def do_GET(self):
        """Обработка GET запросов"""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "model_source": "clearml-registry",
                "model_loaded": self.classifier.classifier is not None
            }).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def do_POST(self):
        """Обработка POST запросов"""
        if self.path != "/predict":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
            return
        
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            
            text = data.get("text", "")
            
            result = self.classifier.predict(text)
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def log_message(self, format, *args):
        """Отключаем лишние логи"""
        pass


def run_server(port=8080, host="127.0.0.1"):
    """Запуск HTTP сервера"""
    classifier = SentimentClassifier()
    RequestHandler.classifier = classifier
    
    server = HTTPServer((host, port), RequestHandler)
    print(f"Сервер запущен на http://{host}:{port}")
    print(f"Endpoint: http://{host}:{port}/predict")
    print(f"Health check: http://{host}:{port}/health")
    print("Нажмите Ctrl+C для остановки")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка сервера...")
        server.shutdown()


if __name__ == "__main__":
    run_server()