"""Foreground supervisor for ONE shared input64 P3+P4+P5 model.

Keep independent single-head queue unchanged. Refuse stale GPU gates and
existing run names. Persist progress and evaluate best-Val on Test after train.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.launch_geometry_v5_heads import ROOT, csv_progress, now

SOURCES = ['experiments/direct_geometry_fusion.py', 'experiments/run_318_fusion_alpha.py',
           'experiments/direct_geometry_monitor.py', 'ultralytics/yolo/data/direct_geometry.py',
           'ultralytics/yolo/data/base.py', 'ultralytics/yolo/data/augment.py',
           'ultralytics/yolo/data/dataset.py', 'ultralytics/yolo/engine/trainer.py',
           'ultralytics/nn/tasks.py', 'ultralytics/yolo/engine/validator.py']


def run(epochs, tag, gate=None, skip_test=False, workers=8, threads=8, seed=0,
        yolo_weight=None, detail_weight=None, archive=None, data=None):
    if epochs < 1 or workers < 0 or threads < 1 or seed < 0 or not tag.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Invalid launch setting')
    if gate is not None:
        audit = json.loads(gate.read_text())
        assert audit['passed'] and set(audit['heads']) == {'high', 'mid', 'low'}
        assert audit['configuration']['geometry_multi_head'] and audit['configuration']['geometry_scatter_mode'] == 'input64'
        for name, expected in audit['code_sha256'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, f'Stale multi-head GPU audit: {name}'
    folder = ROOT / 'experiments/analysis' / f'geometry_v5_input64_multi_e{epochs}_{tag}'
    name = f'best318_multihead_p3p4p5_detecthead_geometry_v5_input64_sum_e{epochs}_lr1e4_{tag}'
    run_dir = ROOT / 'experiments/runs' / name
    if folder.exists() or run_dir.exists():
        raise FileExistsError('Refusing to overwrite/increment experiment name')
    folder.mkdir(parents=True)
    common = ['--multi-head-direct', '--multi-head-scale-aware', '--multi-heads', 'high', 'mid', 'low',
              '--corrected-geometry', '--geometry-input-support', '--scatter-reduce', 'sum',
              '--candidate-conf', '0.2', '--batch', '32', '--imgsz', '640', '--workers', str(workers),
              '--device', '0', '--lr0', '0.0001', '--optimizer', 'AdamW', '--save-period', '-1',
              '--wandb-mode', 'disabled', '--seed', str(seed)]
    for flag, value in [('--yolo-weight', yolo_weight), ('--detail-weight', detail_weight),
                        ('--archive', archive), ('--data', data)]:
        if value is not None:
            common.extend([flag, str(value)])
    base = [sys.executable, '-m', 'experiments.run_318_fusion_alpha']
    command = base + ['train'] + common + ['--epochs', str(epochs), '--name', name]
    env = os.environ.copy()
    env.update(WANDB_MODE='disabled', MPLCONFIGDIR='/tmp/bgd_mpl_multi_input64',
        OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1',
        BGD_CORRECTED_GEOMETRY='1', BGD_INCLUDE_ORI_IMG='0', PYTHONUNBUFFERED='1')
    state = {'started_at': now(), 'phase': 'train', 'epochs': epochs, 'run': name,
        'shared_yolo_and_detail': True, 'independent_fusion_projections': 3, 'one_joint_checkpoint': True,
        'head_selects': ['high', 'mid', 'low'], 'scatter_reduce': 'sum', 'alpha': 'absent',
        'initialization_seed': seed, 'trainer_seed': seed, 'deterministic': True,
        'scatter_patch_hw': {'high': [8, 8], 'mid': [4, 4], 'low': [2, 2]},
        'workers': workers, 'threads': threads,
        'gpu_audit': str(gate) if gate is not None else None, 'command': command,
        'best_weight': str(run_dir / 'weights/best.pt'), 'last_weight': str(run_dir / 'weights/last.pt'),
        'train_log': str(folder / 'multi.train.log'), 'skip_test': skip_test,
        'test_protocol': {'batch': 1, 'precision': 'FP32', 'workers': 4, 'cpu_threads': 4,
                          'candidate_conf': .2, 'final_conf': .5, 'nms_iou': .5},
        'code_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in SOURCES + ['experiments/launch_input64_multi.py', 'experiments/test_multi_geometry.py']}}
    process = None

    def save():
        state['updated_at'] = now()
        state['progress'] = csv_progress(name)
        temporary = folder / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(folder / 'status.json')

    try:
        with Path(state['train_log']).open('w') as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            state['pid'] = process.pid
            save()
            print('INPUT64_MULTI_STARTED ' + json.dumps(state), flush=True)
            while process.poll() is None:
                save()
                time.sleep(5)
        state['train_exit_code'] = process.returncode
        if process.returncode != 0:
            raise RuntimeError(f'Multi-head train failed ({process.returncode}); see {state["train_log"]}')
        assert Path(state['best_weight']).exists() and Path(state['last_weight']).exists()
        state['training_finished_at'] = now()
        state['phase'] = 'train_complete' if skip_test else 'test'
        save()
        if not skip_test:
            # No test-based checkpoint selection or competing training here.
            test_cmd = base + ['val'] + common + ['--checkpoint', state['best_weight'], '--split', 'test',
                '--eval-conf', '0.5', '--eval-iou', '0.5', '--eval-batch', '1', '--workers', '4', '--name', name + '_test']
            test_env = dict(env, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
            state['test_command'] = test_cmd
            state['test_log'] = str(folder / 'multi.test.log')
            save()
            with Path(state['test_log']).open('w') as handle:
                process = subprocess.Popen(test_cmd, cwd=ROOT, env=test_env, stdout=handle, stderr=subprocess.STDOUT)
                state['test_pid'] = process.pid
                while process.poll() is None:
                    save()
                    time.sleep(5)
            state['test_exit_code'] = process.returncode
            if process.returncode != 0:
                raise RuntimeError(f'Multi-head Test failed ({process.returncode}); see {state["test_log"]}')
            state['test_metrics'] = next((json.loads(line[len('EXACT_METRICS '):])
                for line in reversed(Path(state['test_log']).read_text().splitlines()) if line.startswith('EXACT_METRICS ')), None)
            assert state['test_metrics'] is not None, 'Test did not return exact metrics'
        state['phase'], state['finished_at'] = 'complete', now()
        save()
        print('INPUT64_MULTI_COMPLETE ' + str(folder / 'status.json'), flush=True)
    except BaseException as error:
        state['phase'] = 'stopped_by_user' if isinstance(error, KeyboardInterrupt) else 'failed'
        state['error'] = str(error)
        if process is not None and process.poll() is None:
            process.terminate()  # only this supervisor's own child
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--tag', default='r1')
    parser.add_argument('--gate', type=Path, default=None,
                        help='optional matching GPU audit JSON; source hashes are checked when provided')
    parser.add_argument('--skip-test', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--yolo-weight', type=Path, default=None)
    parser.add_argument('--detail-weight', type=Path, default=None)
    parser.add_argument('--archive', type=Path, default=None)
    parser.add_argument('--data', type=Path, default=None)
    args = parser.parse_args()
    run(args.epochs, args.tag, args.gate, args.skip_test, args.workers, args.threads, args.seed,
        args.yolo_weight, args.detail_weight, args.archive, args.data)
