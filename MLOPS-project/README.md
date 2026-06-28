# Лабораторная работа: MLOps с ClearML
## Выполнил Цыкунов Никита
## Social Media Sentiments Analysis 

**Бинарная классификация тональности** текстов из соцсетей.

Используется датасет [Social Media Sentiments Analysis Dataset](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset) (Kaggle). Исходные эмоции (`Joy`, `Sad`, `Excitement` и т.д.) приводятся к двум классам: `positive` / `negative`.

### Что реализовано

| Этап | Скрипт | Назначение |
|------|--------|------------|
| 0 | (Ручками через сайт и терминал) | ClearML Server, SDK, Agent, очередь `students` |
| 1 | `ingest_dataset.py` | Kaggle → CSV → ClearML Dataset |
| 2 | `train.py` | Обучение через Agent, метрики, артефакты |
| 3 | `register_model.py` | Публикация модели в Model Registry |
| 4 | `inference_api.py` | FastAPI endpoint, модель из Registry |
| 5 | `streamlit_app.py` | UI с помощью Streamlit |

### Структура репозитория

```text
.
├── ingest_dataset.py    # этап 1 — датасет в ClearML
├── train.py             # этап 2 — обучение на агенте
├── register_model.py    # этап 3 — Model Registry
├── inference_api.py     # этап 4 — HTTP inference
├── streamlit_app.py     # этап 5 — пользовательский интерфейс
├── requirements.txt
└── social_sentiments.csv  # создаётся после ingest_dataset.py
```

## Как использовать:

### Создание виртуального окружения:

```powershell
python -m venv venv_new
venv_new\Scripts\activate
```

---

### Установка библиотек и всего прочего

```powershell
pip install -r requirements.txt
```

---

## Инфраструктура ClearML

1. Выполнить команду `clearml_init`, перейти по ссылке, зарегистрироваться и указать credentials сервера (по инструкции в терминале).
```powershell
clearml-init
``` 
2. Запустить агента на очереди `students`:

```powershell
clearml-agent daemon --queue students
```

---

## Загрузка Dataset в ClearML

Скрипт `ingest_dataset.py`:

- скачивает датасет через `kagglehub`;
- берёт колонки `Text` и `Sentiment`;
- маппит эмоции в `positive` / `negative`;
- сохраняет подмножество в `social_sentiments.csv`;
- создаёт ClearML Dataset и фиксирует версию.

```powershell
python ingest_dataset.py
```
!Важно - надо скопировать выведенный **Dataset ID** (можно посмотреть ручками в ClearML) и вставить в `train.py` (строка 20):

```python
DATASET_ID = "ваш_dataset_id"
```

---

## Обучение через ClearML Agent

`train.py` создаёт ClearML Task, отправляет его в очередь `students` и логирует:

- гиперпараметры (`C`, `max_iter`, `max_features`, `test_size`);
- метрики `accuracy` и `f1`;
- confusion matrix;
- артефакт `model` (`sentiment_model.pkl`).

Для прогона разных версий модели (два эксперимента) - переключить константу `ACTIVE_RUN` в файле train.py (строки 27-28):

| Параметр | `alpha` | `beta` |
|----------|---------|--------|
| `C` | 1.5 | 0.3 |
| `max_iter` | 250 | 400 |
| `max_features` | 6000 | 12000 |
| `test_size` | 0.25 | 0.2 |

```powershell
# эксперимент 1
# ACTIVE_RUN = "alpha"
python train.py

# эксперимент 2
# ACTIVE_RUN = "beta"
python train.py
```

---

## Model Registry

Выберите лучший Task по метрикам и укажите его ID в `register_model.py` (строка 4):

```python
BEST_TASK_ID = "ваш_task_id"
```

```powershell
python register_model.py
```

Скрипт создаёт `OutputModel`, добавляет теги и публикует модель в **Model Registry**.

---

## Inference Endpoint

Подставьте **Model ID** из Registry в `inference_api.py` (строка 11):

```python
REGISTRY_MODEL_ID = "ваш_model_id"
```

Запуск сервиса:

```powershell
uvicorn inference_api:app --host 127.0.0.1 --port 8000
```

Health-check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Примеры запросов:

```powershell
# positive
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"text":"Absolutely love this product, best purchase ever!"}'

# negative
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"text":"Terrible experience, completely disappointed and frustrated."}'
```

Ответ:

```json
{
  "label": "positive",
  "confidence": 0.57
}
```

---

## Этап 5. Streamlit UI

Поднимаем интерфейс через Streamlit

```powershell
uvicorn inference_api:app --host 127.0.0.1 --port 8000
streamlit run streamlit_app.py
```

Интерфейс: http://localhost:8501
