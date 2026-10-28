import io
import warnings
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from api.inference import load_models, predict_image

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
prediction_lock = Lock()


@asynccontextmanager
async def lifespan(app):
    app.state.classifier, app.state.detector = load_models()
    yield


app = FastAPI(title="BrainScan AI", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(415, "Upload a JPG or PNG image")
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image must be at most 10 MB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as uploaded:
                if uploaded.format not in ("JPEG", "PNG"):
                    raise HTTPException(415, "Upload a JPG or PNG image")
                image = uploaded.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "Invalid or damaged image") from None
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(413, "Image dimensions are too large") from None

    # YOLO's predictor holds mutable state; serialize concurrent requests.
    with prediction_lock:
        return predict_image(image, app.state.classifier, app.state.detector)
