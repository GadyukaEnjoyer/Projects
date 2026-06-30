from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from clearml import Dataset, Task
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

# --- В DATASET_ID вставляем значение из ClearML - можно посмотреть в выдаче после ingest_dataset.py ---
DATASET_ID = "9ef351b3f114498bb9ea00494afa04a1"
REMOTE_QUEUE = "students"
PROJECT = "social-media-sentiment"

# --- Нужно переключить значение ACTIVE_RUN между ---
# --- alpha и beta чтобы выбрать версию эксперимента ---

ACTIVE_RUN = "alpha"
# ACTIVE_RUN = "beta"

RUN_PROFILES = {
    "alpha": {
        "task_name": "tfidf-lr-alpha",
        "C": 1.5,
        "max_iter": 250,
        "max_features": 6000,
        "test_size": 0.25,
    },
    "beta": {
        "task_name": "tfidf-lr-beta",
        "C": 0.3,
        "max_iter": 400,
        "max_features": 12000,
        "test_size": 0.2,
    },
}


def load_clearml_dataset(dataset_id: str) -> pd.DataFrame:
    local_dir = Path(Dataset.get(dataset_id=dataset_id).get_local_copy())
    csv_files = list(local_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"CSV не найден в {local_dir}")
    return pd.read_csv(csv_files[0])


def train_and_evaluate(params: dict, frame: pd.DataFrame):
    features, target = frame["text"], frame["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=params["test_size"],
        random_state=17,
        stratify=target,
    )

    vectorizer = TfidfVectorizer(
        max_features=params["max_features"],
        ngram_range=(1, 2),
        min_df=2,
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    classifier = LogisticRegression(
        C=params["C"],
        max_iter=params["max_iter"],
        class_weight="balanced",
    )
    classifier.fit(x_train_vec, y_train)
    predictions = classifier.predict(x_test_vec)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, pos_label="positive"),
    }

    artifact_path = Path("artifacts") / "sentiment_model.pkl"
    artifact_path.parent.mkdir(exist_ok=True)
    with artifact_path.open("wb") as handle:
        pickle.dump({"model": classifier, "vectorizer": vectorizer}, handle)

    return metrics, artifact_path, y_test, predictions


def log_experiment(
    task: Task,
    metrics: dict,
    y_true: pd.Series,
    y_pred,
    artifact_path: Path,
) -> None:
    logger = task.get_logger()
    logger.report_scalar("accuracy", "validation", metrics["accuracy"], iteration=0)
    logger.report_scalar("f1", "validation", metrics["f1"], iteration=0)

    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(ax=axis, cmap="Blues")
    axis.set_title("Confusion matrix (validation)")
    logger.report_matplotlib_figure("confusion_matrix", "validation", figure, iteration=0)
    plt.close(figure)

    task.upload_artifact("model", artifact_object=str(artifact_path))


def main() -> None:
    profile = RUN_PROFILES[ACTIVE_RUN]
    task = Task.init(
        project_name=PROJECT,
        task_name=profile["task_name"],
        auto_connect_frameworks={"matplotlib": False},
    )
    task.execute_remotely(queue_name=REMOTE_QUEUE)

    hyperparams = task.connect(
        {
            "C": profile["C"],
            "max_iter": profile["max_iter"],
            "max_features": profile["max_features"],
            "test_size": profile["test_size"],
            "dataset_id": DATASET_ID,
        }
    )

    dataframe = load_clearml_dataset(hyperparams["dataset_id"])
    metrics, model_file, y_test, predictions = train_and_evaluate(hyperparams, dataframe)
    log_experiment(task, metrics, y_test, predictions, model_file)

    print(
        f"[{profile['task_name']}] "
        f"accuracy={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}"
    )


if __name__ == "__main__":
    main()
