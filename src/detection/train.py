"""Selected unbalanced four-class 640 configuration from notebook 04."""
PARAMS = dict(epochs=60, patience=15, imgsz=640, batch=8, workers=2,
              optimizer='AdamW', lr0=5e-4, lrf=0.01, cos_lr=True,
              weight_decay=0.0005, warmup_epochs=3.0, box=7.5, cls=0.5, dfl=1.5,
              degrees=10.0, translate=0.05, scale=0.10, fliplr=0.5, flipud=0.0,
              hsv_h=0.015, hsv_s=0.2, hsv_v=0.2, mosaic=0.5, mixup=0.0, seed=42)


def train(output):
    from ultralytics import YOLO, settings
    settings.update({'mlflow': False})
    model = YOLO(str(output / 'yolov8n.pt'))
    model.train(data=str(output / 'data.yaml'), device='cpu', project=str(output),
                name='training', exist_ok=False, plots=False, **PARAMS)
    return output / 'training/weights/best.pt'
