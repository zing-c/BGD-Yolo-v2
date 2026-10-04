"""No-GPU tests for the experiment queue and failure isolation."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import launch_geometry_v5_heads as launcher


class LauncherTests(unittest.TestCase):
    def exercise(self, parallel, fail=False):
        clock, processes = [0], []
        active_peak = [0]

        class Process:
            def __init__(self, command, **kwargs):
                self.command, self.environment = command, kwargs['env']
                self.started, self.returncode, self.terminated = clock[0], None, False
                self.pid = 1000 + len(processes)
                processes.append(self)
                active_peak[0] = max(active_peak[0], sum(p.returncode is None for p in processes))

            def poll(self):
                duration = 4 if fail and self is not processes[0] else 2
                if self.returncode is None and clock[0] - self.started >= duration:
                    self.returncode = 1 if fail and self is processes[0] else 0
                return self.returncode

            def terminate(self):
                self.terminated, self.returncode = True, -15

            def wait(self, timeout=None):
                return self.returncode

        with tempfile.TemporaryDirectory(prefix='bgd_queue_test_') as directory:
            root = Path(directory)
            source = root / 'audit_fixture.txt'
            source.write_text('generated audit fixture\n')
            gate = root / 'experiments/analysis/yolo_finedet_single_input64_audit.json'
            gate.parent.mkdir(parents=True)
            gate.write_text(json.dumps({'passed': True, 'scatter_mode': 'input64',
                'heads': [{'head': head, 'scatter_patch_hw': size}
                          for head, size in [('high', [8, 8]), ('mid', [4, 4]), ('low', [2, 2])]],
                'code_sha256': {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}}))
            with patch.object(launcher, 'ROOT', root), patch.object(launcher, 'SOURCES', [source.name]), \
                 patch.object(launcher.subprocess, 'Popen', Process), \
                 patch.object(launcher.time, 'sleep', lambda _seconds: clock.__setitem__(0, clock[0] + 1)), \
                 patch('builtins.print'):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'failed'):
                        launcher.run(50, 'test', True,
                                     heads=['high', 'mid', 'low'], max_parallel=parallel, workers=4, threads=4,
                                     seed=3187, gate=gate, yolo_weight=root / 'yolo.pt',
                                     detail_weight=root / 'detail.pt', archive=root / 'code.zip',
                                     data=root / 'data.yaml')
                else:
                    launcher.run(50, 'test', True,
                                 heads=['high', 'mid', 'low'], max_parallel=parallel, workers=4, threads=4,
                                 seed=3187, gate=gate, yolo_weight=root / 'yolo.pt',
                                 detail_weight=root / 'detail.pt', archive=root / 'code.zip',
                                 data=root / 'data.yaml')
            status = json.loads((gate.parent / 'geometry_v5_input64_e50_test/status.json').read_text())
            return status, processes, active_peak[0]

    def test_serial_and_parallel_keep_distinct_jobs_and_parameters(self):
        for concurrency in (1, 2, 3):
            status, processes, peak = self.exercise(concurrency)
            self.assertEqual(status['phase'], 'complete')
            self.assertEqual(peak, concurrency)
            self.assertEqual(set(status['jobs']), {'p3', 'p4', 'p5'})
            self.assertEqual(len({job['best_weight'] for job in status['jobs'].values()}), 3)
            self.assertTrue(all(job['train_exit_code'] == 0 for job in status['jobs'].values()))
            for process in processes:
                self.assertIn('--geometry-input-support', process.command)
                self.assertEqual(process.command[process.command.index('--lr0') + 1], '0.0001')
                self.assertEqual(process.command[process.command.index('--seed') + 1], '3187')
                for flag, filename in [('--yolo-weight', 'yolo.pt'), ('--detail-weight', 'detail.pt'),
                                       ('--archive', 'code.zip'), ('--data', 'data.yaml')]:
                    self.assertEqual(Path(process.command[process.command.index(flag) + 1]).name, filename)
                self.assertEqual(process.environment['OMP_NUM_THREADS'], '4')
            self.assertEqual(status['initialization_seed'], 3187)
            self.assertEqual(status['trainer_seed'], 3187)
            self.assertTrue(status['deterministic'])

    def test_failure_stops_only_other_owned_jobs(self):
        status, processes, _ = self.exercise(3, fail=True)
        self.assertEqual(status['phase'], 'failed')
        self.assertFalse(processes[0].terminated)
        self.assertTrue(all(process.terminated for process in processes[1:]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
