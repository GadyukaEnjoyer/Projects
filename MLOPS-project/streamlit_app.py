import time

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8080/predict"
REQUEST_TIMEOUT_SEC = 8

st.set_page_config(page_title="Social Sentiment Lab", page_icon="💬", layout="centered")

st.title("Анализ тональности соцмедиа")
st.markdown(
    "Введите текст поста — предсказание выполняется через **ClearML Serving**."
)

user_text = st.text_area(
    label="Текст поста",
    placeholder="Например: Absolutely love this new update! #happy",
    height=140,
)

if st.button("Predict", type="primary", use_container_width=True):
    cleaned = user_text.strip()
    if not cleaned:
        st.warning("Поле текста не должно быть пустым.")
    else:
        started = time.perf_counter()
        try:
            response = requests.post(
                API_URL,
                json={"text": cleaned},
                timeout=REQUEST_TIMEOUT_SEC,
            )
            response.raise_for_status()
            payload = response.json()
            elapsed_ms = (time.perf_counter() - started) * 1000

            if "error" in payload:
                st.error(f"Ошибка модели: {payload['error']}")
            else:
                label = payload["label"]
                confidence = payload["confidence"]
                style = st.success if label == "positive" else st.error
                style(f"**{label.upper()}** — уверенность {confidence:.1%}")
                st.caption(f"Latency: {elapsed_ms:.1f} ms")

        except requests.exceptions.ConnectionError:
            st.error(
                "ClearML Serving недоступен. "
                "Запустите: `python inference_serving.py`"
            )
        except requests.exceptions.Timeout:
            st.error("Превышено время ожидания ответа от сервиса.")
        except requests.exceptions.HTTPError as exc:
            st.error(f"HTTP-ошибка: {exc.response.status_code}")
        except Exception as exc:
            st.error(f"Не удалось получить предсказание: {exc}")