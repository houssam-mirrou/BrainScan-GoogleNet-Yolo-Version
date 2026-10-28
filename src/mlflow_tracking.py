"""Import saved experiment results; this script does not train or evaluate models."""
import csv
import json
import os
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]


def log_history(path):
    with path.open() as history_file:
        for row in csv.DictReader(history_file):
            metrics = {key: float(value) for key, value in row.items() if key != "epoch"}
            mlflow.log_metrics(metrics, step=int(row["epoch"]))
    mlflow.log_artifact(str(path), "results")


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("BrainScan AI — saved experiments")
    classification_dir = ROOT / "models/classification"
    detection_dir = ROOT / "models/final/yolo"
    tags = {"tracking_type": "retrospective_import", "source": "existing project results"}

    metadata_path = classification_dir / "classification_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    with mlflow.start_run(run_name=metadata["model_name"], tags=tags):
        mlflow.log_params({
            "architecture": metadata["architecture"],
            "training_strategy": metadata["training_strategy"],
            "input_size": metadata["input_size"],
            "preprocessing": metadata["preprocessing"],
            "classes": metadata["classes"],
        })
        mlflow.log_metrics({key: metadata[key] for key in (
            "validation_accuracy", "validation_macro_f1", "test_accuracy", "test_macro_f1",
        )})
        log_history(classification_dir / "inceptionv3_multiscale_v2_history.csv")
        mlflow.log_artifact(str(metadata_path), "results")
        mlflow.log_artifact(str(classification_dir / "inceptionv3_v2_test_report.txt"), "results")
        mlflow.log_artifact(str(ROOT / metadata["model_path"]), "checkpoints")
        print(f"Logged classification run: {mlflow.active_run().info.run_id}")

    metadata_path = detection_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    with mlflow.start_run(run_name=metadata["model"], tags=tags):
        mlflow.log_params({key: metadata[key] for key in (
            "model", "image_size", "classes", "selected_by", "checkpoint_sha256",
        )})
        mlflow.log_metrics({f"test_{key}": value for key, value in metadata["test_metrics"].items()})
        for name in ("metadata.json", "test_metrics.csv", "test_per_class.csv"):
            mlflow.log_artifact(str(detection_dir / name), "results")
        mlflow.log_artifact(str(detection_dir / "best.pt"), "checkpoints")
        print(f"Logged detection run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
