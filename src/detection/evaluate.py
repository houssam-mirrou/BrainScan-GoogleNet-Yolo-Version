from src.common.validation import CLASSES


def evaluate(checkpoint, output):
    from ultralytics import YOLO
    model = YOLO(str(checkpoint))
    if model.names != dict(enumerate(CLASSES)):
        raise ValueError('Candidate detection class mapping changed')
    result = model.val(data=str(output / 'data.yaml'), split='val', imgsz=640,
                       batch=8, device='cpu', plots=False, save_json=False,
                       project=str(output), name='evaluation')
    metrics = dict(precision=float(result.box.mp), recall=float(result.box.mr),
                   map50=float(result.box.map50), map50_95=float(result.box.map))
    per_class = {}
    for index, class_id in enumerate(result.box.ap_class_index):
        name = CLASSES[int(class_id)]
        values = result.box.class_result(index)
        per_class[name] = float(values[3])
        for key, value in zip(('precision', 'recall', 'map50', 'map50_95'), values):
            metrics[f'{name.replace(" ", "_")}_{key}'] = float(value)
    if set(per_class) != set(CLASSES):
        raise ValueError('Validation did not evaluate every class')
    metrics['tumor_only_map50_95'] = sum(per_class[c] for c in CLASSES if c != 'No Tumor') / 3
    return metrics
