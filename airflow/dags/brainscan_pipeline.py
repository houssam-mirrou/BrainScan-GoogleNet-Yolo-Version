"""Manual, sequential experiment workflow; model work runs in its own Python env."""
from datetime import datetime, timezone
import hashlib
import subprocess

from airflow import DAG
from airflow.operators.python import PythonOperator


def run_stage(kind, stage, **context):
    run_id = hashlib.sha256(context['run_id'].encode()).hexdigest()[:24]
    subprocess.run(['/opt/ml/bin/python', '-m', 'src.pipeline', kind, stage,
                    '--run-id', run_id], cwd='/app', check=True)


with DAG(
    'brainscan_pipeline', start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None, catchup=False, max_active_runs=1, max_active_tasks=1,
    default_args={'retries': 0}, tags=['brainscan', 'manual', 'training'],
    description='Validated data → candidate checkpoints; deployment stays manual.',
) as dag:
    for kind in ('classification', 'detection'):
        previous = None
        for stage in ('validate', 'preprocess', 'train', 'evaluate', 'log', 'save'):
            task = PythonOperator(task_id=f'{kind}_{stage}', python_callable=run_stage,
                                  op_kwargs={'kind': kind, 'stage': stage})
            if previous is not None:
                previous >> task
            previous = task
