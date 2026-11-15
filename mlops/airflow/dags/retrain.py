"""Airflow DAG — periodic retraining and drift-gated redeployment (CMMI ML5 OID/QPM).

Schedule:
    weekly @ 02:00 Asia/Tehran, plus an ad-hoc trigger when drift alarm fires.

Pipeline steps (each is a task; failures raise a CAR-eligible alert):
    1. fetch_data          — scripts/download_data.py (catalogued sources)
    2. prepare_data        — scripts/prepare_data.py
    3. train               — mlops/pipelines/train_pipeline.py
    4. evaluate            — mlops/pipelines/eval_pipeline.py (acceptance gates)
    5. drift_check         — mlops/drift/drift_report.py
    6. deploy_decision     — branch: promote (deploy_pipeline) or hold (CAR)
    7. notify              — report status to stakeholders
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

REPO = os.environ.get("WELD_REPO", "/opt/weld-defect")
PY = os.environ.get("WELD_PYTHON", "python")

default_args = {
    "owner": "petro-pala-toos",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": True,
    "email": ["r&d@petropalatoos.ir"],
}

dag = DAG(
    dag_id="weld_defect_retrain",
    default_args=default_args,
    description="Weekly retrain + drift-gated redeploy for the weld-defect ensemble (CMMI ML5).",
    schedule="0 2 * * 0",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["weld-defect", "mlops", "cmmi-ml5"],
    default_view="graph",
)

t_fetch = BashOperator(task_id="fetch_data", bash_command=f"cd {REPO} && {PY} scripts/download_data.py --sources sate_mazandaran_gas || true", dag=dag)
t_prepare = BashOperator(task_id="prepare_data", bash_command=f"cd {REPO} && {PY} scripts/prepare_data.py --raw data/raw/sate_mazandaran_gas --processed data/processed", dag=dag)
t_train = BashOperator(task_id="train", bash_command=f"cd {REPO} && {PY} mlops/pipelines/train_pipeline.py --config configs/train.yaml", dag=dag)
t_eval = BashOperator(task_id="evaluate", bash_command=f"cd {REPO} && {PY} mlops/pipelines/eval_pipeline.py --config configs/eval.yaml", dag=dag)
t_drift = BashOperator(task_id="drift_check",
                       bash_command=f"cd {REPO} && {PY} mlops/drift/drift_report.py "
                                    "--reference metrics/baselines/reference_profile.json "
                                    "--current metrics/latest/current_profile.json",
                       dag=dag)


def _branch(**ctx):
    import json
    from pathlib import Path
    p = Path(f"{REPO}/metrics/eval_history.jsonl")
    if not p.exists():
        return "hold_and_car"
    last = json.loads(p.read_text(encoding="utf-8").strip().splitlines()[-1])
    return "deploy" if last.get("acceptance_passed") else "hold_and_car"


t_branch = BranchPythonOperator(task_id="deploy_decision", python_callable=_branch, dag=dag, provide_context=True)
t_deploy = BashOperator(task_id="deploy", bash_command=f"cd {REPO} && {PY} mlops/pipelines/deploy_pipeline.py --canary", dag=dag)
t_hold = BashOperator(task_id="hold_and_car", bash_command=f"cd {REPO} && echo 'HOLD — acceptance failed; CAR seeded by eval_pipeline.' >> metrics/retrain_log.txt", dag=dag)
t_notify = PythonOperator(task_id="notify",
                          python_callable=lambda **ctx: print("retrain DAG finished; see metrics/ for QPM/OPP feed."),
                          dag=dag)

t_fetch >> t_prepare >> t_train >> t_eval >> t_drift >> t_branch
t_branch >> [t_deploy, t_hold]
[t_deploy, t_hold] >> t_notify
