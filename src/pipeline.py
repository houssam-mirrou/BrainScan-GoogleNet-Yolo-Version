"""Small stage commands shared by Airflow and local checks."""
import argparse
import importlib
import json
import math

from src.common.validation import ROOT, CLASSES, validate, digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def run(kind, stage, run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('Use an alphanumeric run ID (hyphens/underscores allowed)')
    base = ROOT / 'models/candidates'
    output = base / run_id / kind
    if not output.resolve().is_relative_to(base.resolve()) or base.is_symlink():
        raise ValueError('Candidate output must stay inside models/candidates')
    if stage == 'validate':
        report = validate(kind)
        output.mkdir(parents=True, exist_ok=False)
        write(output / 'validation.json', report)
        return
    report = json.loads((output / 'validation.json').read_text())
    if stage == 'preprocess':
        # Reuse audited data; only relocate the absolute YOLO path for Docker.
        if kind == 'detection':
            import yaml
            dataset = ROOT / 'data/processed/detection_4class'
            (output / 'data.yaml').write_text(yaml.safe_dump(dict(
                path=str(dataset), train='images/train', val='images/val', names=dict(enumerate(CLASSES)))))
        write(output / 'preprocessing.json', dict(reused_processed_data=True,
              image_size=224 if kind == 'classification' else 640, classes=CLASSES))
    elif stage == 'train':
        # Detect edits between validation and training before loading any model.
        if validate(kind) != report:
            raise ValueError('Dataset changed after validation')
        module = importlib.import_module(f'src.{kind}.train')
        checkpoint = module.train(output)
        write(output / 'training.json', dict(checkpoint=str(checkpoint.relative_to(output)), parameters=module.PARAMS))
    elif stage == 'evaluate':
        training = json.loads((output / 'training.json').read_text())
        metrics = importlib.import_module(f'src.{kind}.evaluate').evaluate(output / training['checkpoint'], output)
        if not metrics or not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError('Evaluation returned missing/non-finite metrics')
        write(output / 'metrics.json', metrics)
    elif stage == 'log':
        import os
        import mlflow
        training = json.loads((output / 'training.json').read_text())
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        mlflow.set_experiment(f'BrainScan AI — {kind} candidates')
        with mlflow.start_run(run_name=f'{run_id}-{kind}') as tracking_run:
            mlflow.set_tags(dict(model_type=kind, selection_split='val', promotion='manual'))
            mlflow.log_params(training['parameters'])
            mlflow.log_param('architecture', 'InceptionV3_MultiScale_V2' if kind == 'classification' else 'YOLOv8n')
            mlflow.log_param('dataset_version', report['dataset_version'])
            mlflow.log_param('classes', json.dumps(CLASSES))
            mlflow.log_metrics(json.loads((output / 'metrics.json').read_text()))
            # Explicit allowlist: never upload images, YOLO batch plots or datasets.
            for name in ('validation.json', 'preprocessing.json', 'training.json', 'metrics.json', 'history.csv', 'confusion_matrix.csv', 'training/results.csv', 'training/args.yaml'):
                if (output / name).is_file():
                    mlflow.log_artifact(str(output / name), 'results')
            mlflow.log_artifact(str(output / training['checkpoint']), 'checkpoints')
            write(output / 'mlflow.json', dict(run_id=tracking_run.info.run_id))
    elif stage == 'save':
        training = json.loads((output / 'training.json').read_text())
        checkpoint = output / training['checkpoint']
        write(output / 'candidate.json', dict(model_type=kind, checkpoint=training['checkpoint'],
              sha256=digest(checkpoint), metrics=json.loads((output / 'metrics.json').read_text()),
              mlflow=json.loads((output / 'mlflow.json').read_text()), promotion='manual'))
    print(f'{kind}: {stage} completed: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['classification', 'detection'])
    parser.add_argument('stage', choices=['validate', 'preprocess', 'train', 'evaluate', 'log', 'save'])
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run(args.kind, args.stage, args.run_id)
