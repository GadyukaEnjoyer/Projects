from __future__ import annotations

from pathlib import Path

import kagglehub
import pandas as pd
from clearml import Dataset

# --- Конфиг ---
CLEARML_PROJECT = "social-media-sentiment"
DATASET_NAME = "social_media_sentiments"
KAGGLE_SLUG = "kashishparmar02/social-media-sentiments-analysis-dataset"
OUTPUT_FILE = Path("social_sentiments.csv")
ROW_LIMIT = 3000

POSITIVE_EMOTIONS = {
    "Positive", "Joy", "Excitement", "Happy", "Contentment", "Gratitude",
    "Hopeful", "Elation", "Love", "Admiration", "Pride", "Amusement",
    "Affection", "Bliss", "Delight", "Enthusiasm", "Euphoria", "Optimism",
    "Positivity", "Radiance", "Serenity", "Triumph", "Wonder", "Celebration",
    "Confidence", "Ecstasy", "Harmony", "Inspiration", "Passion", "Relief",
    "Satisfaction", "Thrill", "Warmth",
}

NEGATIVE_EMOTIONS = {
    "Negative", "Sad", "Bad", "Despair", "Loneliness", "Embarrassed",
    "Anger", "Fear", "Hate", "Disgust", "Depression", "Anxiety", "Grief",
    "Shame", "Guilt", "Jealousy", "Regret", "Sorrow", "Stress", "Worry",
    "Frustration", "Disappointment", "Dread", "Horror", "Melancholy",
    "Agony", "Contempt", "Distress", "Heartbreak", "Hopelessness",
    "Isolation", "Outrage", "Panic", "Rage", "Remorse", "Terror", "Tragedy",
}


def fetch_kaggle_archive() -> Path:
    """Скачивает архив с Kaggle и возвращает путь к каталогу."""
    root = Path(kagglehub.dataset_download(KAGGLE_SLUG))
    csv_path = root / "sentimentdataset.csv"
    if not csv_path.exists():
        candidates = list(root.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"CSV не найден в {root}")
        csv_path = candidates[0]
    return csv_path


def emotion_to_binary(raw_sentiment: str) -> str | None:
    """Приводит эмоцию из датасета к бинарной метке positive/negative."""
    label = raw_sentiment.strip()
    if label in POSITIVE_EMOTIONS:
        return "positive"
    if label in NEGATIVE_EMOTIONS:
        return "negative"

    lowered = label.lower()
    negative_hints = (
        "sad", "bad", "fear", "anger", "hate", "lonely", "despair", "grief",
        "shame", "pain", "stress", "worry", "negative", "depress", "anxiet",
        "misery", "regret", "disappoint", "frustrat", "horror", "terror",
        "panic", "rage", "tragedy", "unhappy", "upset", "embarrass",
    )
    positive_hints = (
        "joy", "happy", "love", "excit", "gratitud", "hope", "posit",
        "delight", "bliss", "cheer", "euphor", "optim", "pride", "amuse",
        "celebr", "confiden", "ecstat", "inspir", "passion", "relief",
        "satisf", "thrill", "admire", "content", "elation", "seren",
        "triumph", "wonder", "radianc",
    )

    if any(token in lowered for token in negative_hints):
        return "negative"
    if any(token in lowered for token in positive_hints):
        return "positive"
    return None


def build_training_frame(source_csv: Path) -> pd.DataFrame:
    """Формирует таблицу text + label для обучения классификатора."""
    raw_df = pd.read_csv(source_csv)
    raw_df["Sentiment"] = raw_df["Sentiment"].astype(str).str.strip()
    raw_df["label"] = raw_df["Sentiment"].map(emotion_to_binary)

    prepared = (
        raw_df.dropna(subset=["label"])
        .rename(columns={"Text": "text"})
        [["text", "label"]]
        .drop_duplicates(subset=["text"])
        .head(ROW_LIMIT)
        .reset_index(drop=True)
    )
    return prepared


def push_to_clearml(local_csv: Path) -> Dataset:
    """Создаёт версию датасета в ClearML и загружает файл."""
    dataset = Dataset.create(
        dataset_name=DATASET_NAME,
        dataset_project=CLEARML_PROJECT,
    )
    dataset.add_files(str(local_csv))
    dataset.finalize(auto_upload=True)
    return dataset


def main() -> None:
    source = fetch_kaggle_archive()
    frame = build_training_frame(source)
    frame.to_csv(OUTPUT_FILE, index=False)

    print(f"Подготовлено записей: {len(frame)}")
    print(frame["label"].value_counts().to_string())

    clearml_dataset = push_to_clearml(OUTPUT_FILE)
    print(f"Dataset ID: {clearml_dataset.id}")


if __name__ == "__main__":
    main()
