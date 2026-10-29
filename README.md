# BrainScan AI

A research prototype for brain MRI classification and detection. **Not a clinically validated diagnostic tool.**

## Run

Requires Docker with Compose and the existing checkpoints:

- `models/classification/inceptionv3_multiscale_v2.keras`
- `models/final/yolo/best.pt`

```bash
docker compose up --build -d
docker compose ps
```

Open Streamlit at http://localhost:8501, FastAPI documentation at http://localhost:8000/docs, and MLflow at http://localhost:5000. Upload a JPG/PNG MRI (up to 10 MB), then click **Analyze**.

Models are mounted read-only. Inference runs on CPU. The API loads both models once during startup; first startup includes model loading. Build contexts exclude checkpoints, datasets, notebooks, and virtual environments. Optional checkpoint paths are documented in `.env.example` (copy to `.env` to override).

```bash
docker compose logs api
docker compose down
```

MLflow's named volume persists through `docker compose down`. `docker compose down -v` deletes its stored runs and artifacts.

## API

`GET /health` returns readiness after models load. `POST /predict` accepts multipart field `file`:

```bash
curl -F 'file=@/path/to/mri.jpg' http://localhost:8000/predict
```

The response contains `classification` (predicted class, confidence, class probabilities), `detections` (class ID/name, confidence, `bbox`, `is_tumor`), and original image dimensions. Confidence/probabilities are between 0 and 1. Boxes use `[x1, y1, x2, y2]` in original image pixels. Empty detections do not establish absence of disease. **No Tumor boxes represent annotated non-tumor regions, not tumors.**

The classifier receives bilinearly resized 224×224 RGB float32 pixels in the 0–255 range. Its saved Rescaling layer performs normalization. Class order is Glioma, Meningioma, No Tumor, Pituitary. The notebooks select the frozen Multi-Scale V2 checkpoint; the fine-tuned checkpoint remains a comparison artifact. Keras must support the saved RandomGaussianBlur augmentation layer. Loading uses `compile=False` and inference disables augmentation. YOLO handles its own 640-pixel letterboxing and normalization.

## Import saved experiments into MLflow

After MLflow becomes healthy:

```bash
docker compose exec mlflow python src/mlflow_tracking.py
```

This creates two retrospectively imported runs with existing checkpoints, result files, and recorded metrics. It does not retrain, reevaluate, or claim live tracking of past training. Running it again creates two new runs. The classifier training-history CSV is logged by epoch. Summary metrics come from `classification_metadata.json`; its saved test report is also attached. YOLO test metrics and per-class CSVs come from `models/final/yolo/`.

Recorded classification test accuracy: 0.8910; macro F1: 0.8963. YOLO test mAP50: 0.9531269851; mAP50–95: 0.7254863245. Metrics retain their source precision and interpretation.

## Files

- `api/main.py`: startup, health, image validation, prediction endpoint.
- `api/inference.py`: checkpoint loading and both model predictions.
- `streamlit/app.py`: upload, API request, results and bounding boxes.
- `src/mlflow_tracking.py`: saved-experiment import.
- `docker/`, `docker-compose.yml`, `pyproject.toml`: Python 3.12 services and dependencies.

Existing notebooks, datasets and selected models are preserved. Streamlit never loads models; MLflow is separate from inference.

## Validation performed

Built and started all three Docker services with Python 3.12. Tested `/health`, `/docs`, and real `/predict` requests using existing MRIs from each class; checked probability sums, class mappings, bounding-box bounds, grayscale/RGBA PNGs, invalid/oversized uploads, and concurrent requests. Streamlit's AppTest exercised the real upload and Analyze widgets against FastAPI and verified the preview, confidence, probability chart, annotated image, detection table, and invalid-upload error. This is application integration testing, not a new model accuracy evaluation.

Both MLflow imports completed; saved metrics and downloaded checkpoint hashes were checked, and runs/artifacts survived an MLflow restart. `pip check` passed in all services. The tested API uses TensorFlow CPU 2.21.0, Keras 3.15.1, and Ultralytics 8.4.191; Streamlit 1.66.1 and MLflow 3.17.2 were tested.

## Airflow experiments

`docker compose up --build -d` also starts Airflow at http://localhost:8080.
This local development service uses Airflow 2.11/Python 3.12, SQLite and the
SequentialExecutor. Metadata, logs and candidate checkpoints persist in named
volumes. Find the generated `admin` password with:

```bash
docker compose exec airflow cat /opt/airflow/state/standalone_admin_password.txt
docker compose exec airflow airflow dags list-import-errors
```

Unpause `brainscan_pipeline` and click **Trigger DAG** when ready for real training.
There is no automatic schedule. The Graph view shows each model's validation →
preprocessing → training → validation-set evaluation → MLflow logging → candidate
registration tasks. Click a task to inspect its logs and errors. A failed task
blocks its downstream tasks. Only one task/run executes at a time; training uses
CPU in this development image. ML dependencies run in `/opt/ml`, separate from
Airflow. Training is expensive even on CPU; do not trigger the DAG as a smoke test.

The existing prepared classification and **unbalanced four-class** detection
splits are reused. Manifests, image hashes, class mappings and YOLO boxes must
validate first. New or changed images require reviewed labels and updated
manifests before acceptance. No Tumor retains its annotated boxes in this selected
detector; the older three-class preparation script is deliberately not invoked.
Classification continues the selected frozen V2 `.keras` checkpoint with a fresh
optimizer; detection starts from pretrained `yolov8n.pt` (downloaded on first training).
Notebook training parameters are preserved. Test images are audited for leakage,
but never used for automated evaluation or selection.

Airflow sends metrics, parameters, dataset fingerprint and allowlisted artifacts
through `http://mlflow:5000` to the existing MLflow server. MRI images and datasets
are not uploaded. Each run writes under `/app/models/candidates/<run-hash>/<model>`
in the `airflow_candidates` volume. `candidate.json` records metrics, checkpoint
hash and MLflow run ID after successful logging. Final models are mounted read-only;
promotion and deployment remain manual. Download checkpoints from MLflow or copy
from the container with `docker compose cp`. Removing volumes with `down -v`
also deletes Airflow history and candidates.

Lightweight checks (unique run ID required; neither command trains):

```bash
docker compose exec airflow /opt/ml/bin/python -m src.pipeline classification validate --run-id audit-001
docker compose exec airflow /opt/ml/bin/python -m src.pipeline classification preprocess --run-id audit-001
docker compose exec airflow /opt/ml/bin/python -m src.pipeline detection validate --run-id audit-001
docker compose exec airflow /opt/ml/bin/python -m src.pipeline detection preprocess --run-id audit-001
docker compose exec airflow /opt/ml/bin/python -c "import urllib.request; print(urllib.request.urlopen('http://mlflow:5000/health').status)"
```

The same CLI exposes `train`, `evaluate`, `log`, `save` for each model; the DAG
runs these in order. A new run ID creates a separate experiment directory.
