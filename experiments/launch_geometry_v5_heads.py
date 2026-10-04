"""Queue independent P3/P4/P5 jobs; evaluate only after all training finishes.

Foreground supervisor: do not nohup/detach. Abort our other child if one fails,
keep logs/checkpoints, and write machine-readable status for progress queries.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['experiments/run_318_fusion_alpha.py', 'experiments/direct_geometry_fusion.py',
           'experiments/direct_geometry_monitor.py',
           'ultralytics/yolo/data/direct_geometry.py', 'ultralytics/yolo/data/base.py',
           'ultralytics/yolo/data/augment.py', 'ultralytics/yolo/data/dataset.py',
           'ultralytics/nn/tasks.py', 'ultralytics/yolo/engine/validator.py',
           'ultralytics/yolo/engine/trainer.py', 'experiments/launch_geometry_v5_heads.py']


def now():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()


def csv_progress(run_name):
    path = ROOT / 'experiments/runs' / run_name / 'results.csv'
    if not path.exists():
        return None
    with path.open() as handle:
        rows = [{key.strip(): value.strip() for key, value in row.items() if key is not None and value is not None}
                for row in csv.DictReader(handle)]
    if not rows:
        return None
    return {'completed_epochs': len(rows), 'last': rows[-1],
            'best_val_map50': max(float(row['metrics/mAP50(B)']) for row in rows)}


def run(epochs, tag, skip_test, heads=('mid', 'low'), max_parallel=2, workers=8, threads=8,
        seed=0, gate=None, yolo_weight=None, detail_weight=None, archive=None, data=None):
    heads = list(dict.fromkeys(heads))
    labels = {'high': 'p3', 'mid': 'p4', 'low': 'p5'}
    if epochs < 1 or not 1 <= max_parallel <= 3 or workers < 0 or threads < 1 or seed < 0:
        raise ValueError('Invalid epoch/concurrency/worker/thread setting')
    if not tag.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Use an alphanumeric run tag')
    if gate is not None:
        report = json.loads(gate.read_text())
        assert report['passed'] and {row['head'] for row in report['heads']} >= set(heads)
        assert report['scatter_mode'] == 'input64'
        expected_sizes = {'high': [8, 8], 'mid': [4, 4], 'low': [2, 2]}
        assert all(row['scatter_patch_hw'] == expected_sizes[row['head']] for row in report['heads'])
        assert report.get('code_sha256'), 'Runs need a source-matched GPU audit'
        for name, expected in report['code_sha256'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, f'Stale GPU audit: {name}'
    version = 'geometry_v5_input64'
    folder = ROOT / 'experiments/analysis' / f'{version}_e{epochs}_{tag}'
    if folder.exists():
        raise FileExistsError(f'Refusing to overwrite supervisor artifacts: {folder}')
    folder.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update({'WANDB_MODE': 'disabled', 'MPLCONFIGDIR': '/tmp/bgd_mpl_geometry',
                        'OMP_NUM_THREADS': str(threads), 'MKL_NUM_THREADS': str(threads), 'OPENBLAS_NUM_THREADS': '1',
                        'PYTHONUNBUFFERED': '1', 'BGD_CORRECTED_GEOMETRY': '1', 'BGD_INCLUDE_ORI_IMG': '0'})
    base = [sys.executable, '-m', 'experiments.run_318_fusion_alpha']
    common = ['--corrected-direct-fusion', '--corrected-geometry', '--scatter-reduce', 'sum',
              '--candidate-conf', '0.2', '--batch', '32', '--imgsz', '640', '--workers', str(workers),
              '--device', '0', '--lr0', '0.0001', '--optimizer', 'AdamW',
              '--save-period', '-1', '--wandb-mode', 'disabled', '--seed', str(seed)]
    for flag, value in [('--yolo-weight', yolo_weight), ('--detail-weight', detail_weight),
                        ('--archive', archive), ('--data', data)]:
        if value is not None:
            common.extend([flag, str(value)])
    common.append('--geometry-input-support')
    state = {'started_at': now(), 'phase': 'train', 'epochs': epochs,
             'initialization_seed': seed, 'trainer_seed': seed, 'deterministic': True,
             'independent_weights': True,
             'scatter_mode': 'input64',
             'scatter_patch_hw': {labels[head]: {'high': [8, 8], 'mid': [4, 4], 'low': [2, 2]}[head]
                                  for head in heads},
             'max_parallel': max_parallel, 'workers_per_job': workers, 'threads_per_job': threads,
             'gpu_audit': str(gate) if gate is not None else None, 'skip_test': skip_test,
             'code_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
             'jobs': {}}
    processes, handles = {}, []

    def save():
        state['updated_at'] = now()
        for label, job in state['jobs'].items():
            job['progress'] = csv_progress(job['run'])
        temporary = folder / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(folder / 'status.json')

    try:
        for head in heads:
            label = labels[head]
            name = f'best318_{head}_{label}_detecthead_{version}_sum_e{epochs}_lr1e4_{tag}'
            if (ROOT / 'experiments/runs' / name).exists():
                raise FileExistsError(f'Refusing automatic run-name increment: {name}')
            log = folder / f'{label}.train.log'
            command = base + ['train'] + common + ['--head-select', head, '--epochs', str(epochs), '--name', name]
            state['jobs'][label] = {'run': name, 'head': head, 'pid': None, 'phase': 'queued',
                                   'command': command, 'log': str(log),
                                   'best_weight': str(ROOT / 'experiments/runs' / name / 'weights/best.pt'),
                                   'last_weight': str(ROOT / 'experiments/runs' / name / 'weights/last.pt')}
        save()
        print('GEOMETRY_JOBS_QUEUED ' + json.dumps(state), flush=True)
        pending = list(state['jobs'])
        while pending or any(process.poll() is None for process in processes.values()):
            for label, process in processes.items():
                code = process.poll()
                if code is not None and code != 0:
                    raise RuntimeError(f'{label} training failed ({code}); see {state["jobs"][label]["log"]}')
                if code == 0 and state['jobs'][label]['phase'] != 'train_complete':
                    state['jobs'][label].update({'phase': 'train_complete', 'train_exit_code': 0,
                                                'training_finished_at': now()})
            active = sum(process.poll() is None for process in processes.values())
            while pending and active < max_parallel:
                label = pending.pop(0)
                job = state['jobs'][label]
                handle = Path(job['log']).open('w')
                handles.append(handle)
                process = subprocess.Popen(job['command'], cwd=ROOT, env=environment,
                                           stdout=handle, stderr=subprocess.STDOUT)
                processes[label] = process
                job.update({'pid': process.pid, 'phase': 'train', 'training_started_at': now()})
                active += 1
                print('GEOMETRY_JOB_STARTED ' + json.dumps({'head': label, 'pid': process.pid}), flush=True)
            save()
            time.sleep(5)
        for label, process in processes.items():
            state['jobs'][label]['train_exit_code'] = process.returncode
            state['jobs'][label]['phase'] = 'train_complete'
            if process.returncode != 0:
                raise RuntimeError(f'{label} failed ({process.returncode})')
        state['training_finished_at'] = now()
        state['phase'] = 'train_complete' if skip_test else 'test'
        save()
        if not skip_test:
            # Never benchmark a test process against any training job.
            for label, job in state['jobs'].items():
                log = folder / f'{label}.test.log'
                command = base + ['val'] + common + ['--checkpoint', job['best_weight'], '--split', 'test',
                          '--eval-conf', '0.5', '--eval-iou', '0.5', '--eval-batch', '1',
                          '--head-select', job['head'],
                          '--name', job['run'] + '_test']
                with log.open('w') as handle:
                    result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
                if result.returncode != 0:
                    raise RuntimeError(f'{label} final test failed ({result.returncode}); see {log}')
                job['test_log'] = str(log)
                job['test_metrics'] = next((json.loads(line[len('EXACT_METRICS '):])
                    for line in reversed(log.read_text().splitlines()) if line.startswith('EXACT_METRICS ')), None)
                if job['test_metrics'] is None:
                    raise RuntimeError(f'{label} test produced no EXACT_METRICS record')
                job['phase'] = 'complete'
                save()
        state['phase'] = 'complete'
        state['finished_at'] = now()
        save()
        print('GEOMETRY_JOBS_COMPLETE ' + str(folder / 'status.json'), flush=True)
    except BaseException as error:
        state['phase'] = 'stopped_by_user' if isinstance(error, KeyboardInterrupt) else 'failed'
        if isinstance(error, KeyboardInterrupt):
            state['stopped_at'] = now()
        state['error'] = str(error)
        for process in processes.values():
            if process.poll() is None:
                process.terminate()  # only the supervisor's own training jobs
        for process in processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        save()
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--tag', default='r1')
    parser.add_argument('--skip-test', action='store_true')
    parser.add_argument('--heads', nargs='+', choices=['high', 'mid', 'low'], default=['mid', 'low'])
    parser.add_argument('--max-parallel', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gate', type=Path, default=None,
                        help='optional matching GPU audit JSON; source hashes are checked when provided')
    parser.add_argument('--yolo-weight', type=Path, default=None)
    parser.add_argument('--detail-weight', type=Path, default=None)
    parser.add_argument('--archive', type=Path, default=None)
    parser.add_argument('--data', type=Path, default=None)
    args = parser.parse_args()
    run(args.epochs, args.tag, args.skip_test,
        args.heads, args.max_parallel, args.workers, args.threads, args.seed, args.gate,
        args.yolo_weight, args.detail_weight, args.archive, args.data)
