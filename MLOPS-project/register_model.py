from clearml import OutputModel, Task

# --- В BEST_TASK_ID вставляем значение лучшего эксперимента (именно TASK_ID - можно посмотреть в выдаче train.py) ---
BEST_TASK_ID = "68d5598295ca4d4581ef68b3e68b471a"

MODEL_NAME = "social-sentiment-classifier"
MODEL_TAGS = ["social-media", "tfidf", "logistic-regression", "binary-sentiment"]


def publish_from_task(task_id: str) -> OutputModel:
    source_task = Task.get_task(task_id=task_id)
    artifact_local_path = source_task.artifacts["model"].get_local_copy()

    registered = OutputModel(
        task=source_task,
        name=MODEL_NAME,
        framework="scikit-learn",
        tags=MODEL_TAGS,
    )
    registered.update_weights(weights_filename=artifact_local_path)
    registered.publish()
    return registered


def main() -> None:
    model = publish_from_task(BEST_TASK_ID)
    print(f"Model ID: {model.id}")
    print("Модель опубликована в Model Registry.")


if __name__ == "__main__":
    main()
