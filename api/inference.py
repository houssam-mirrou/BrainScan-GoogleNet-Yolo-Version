import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


def load_models():
    classifier_path = Path(os.getenv(
        "CLASSIFIER_PATH",
        str(ROOT / "models/classification/inceptionv3_multiscale_v2.keras"),
    ))
    detector_path = Path(os.getenv(
        "DETECTOR_PATH", str(ROOT / "models/final/yolo/best.pt"),
    ))
    for path in (classifier_path, detector_path):
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

    classifier = tf.keras.models.load_model(classifier_path, compile=False)
    detector = YOLO(str(detector_path))
    if classifier.input_shape[1:] != (224, 224, 3) or classifier.output_shape[-1] != 4:
        raise ValueError("Expected a 224x224 RGB classifier with four outputs")
    if [detector.names[i] for i in range(4)] != CLASSES:
        raise ValueError(f"Unexpected detector class mapping: {detector.names}")
    return classifier, detector


def predict_image(image, classifier, detector):
    # Match image_dataset_from_directory: bilinear resize, float32, 0–255.
    # The saved classifier already includes InceptionV3 normalization.
    pixels = tf.convert_to_tensor(np.asarray(image), dtype=tf.float32)
    resized = tf.image.resize(pixels, (224, 224), method="bilinear")
    probabilities = classifier(resized[None, ...], training=False).numpy()[0]
    class_id = int(np.argmax(probabilities))

    # Ultralytics handles letterboxing and normalization itself. PIL means RGB.
    result = detector.predict(image, imgsz=640, device="cpu", verbose=False)[0]
    detections = []
    for box in result.boxes:
        detected_id = int(box.cls.item())
        detections.append({
            "class_id": detected_id,
            "class_name": CLASSES[detected_id],
            "confidence": float(box.conf.item()),
            "bbox": [float(value) for value in box.xyxy[0].tolist()],
            "is_tumor": detected_id != 2,
        })

    return {
        "classification": {
            "predicted_class": CLASSES[class_id],
            "confidence": float(probabilities[class_id]),
            "probabilities": {
                name: float(probabilities[i]) for i, name in enumerate(CLASSES)
            },
        },
        "detections": detections,
        "image": {"width": image.width, "height": image.height},
        "bbox_format": "xyxy_original_image_pixels",
        "notice": "No Tumor boxes mark annotated non-tumor regions, not tumors.",
    }
