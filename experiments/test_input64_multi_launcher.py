"""Non-GPU checks for shared-model launch parameters and owned-child cleanup."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import launch_input64_multi as launcher


class MultiLauncherTests(unittest.TestCase):
    def exercise(self, fail=False):
        clock, processes = [0], []
        with tempfile.TemporaryDirectory(prefix='bgd_multi_queue_test_') as directory:
            root = Path(directory)
            source_names = ['fixture.txt', 'experiments/launch_input64_multi.py', 'experiments/test_multi_geometry.py']
            for name in source_names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('generated fixture\n')
            gate = root / 'gate.json'
            gate.write_text(json.dumps({'passed': True, 'heads': ['high', 'mid', 'low'],
                'configuration': {'geometry_multi_head': True, 'geometry_scatter_mode': 'input64'},
                'code_sha256': {'fixture.txt': hashlib.sha256((root / 'fixture.txt').read_bytes()).hexdigest()}}))

            class Process:
                def __init__(self, command, **kwargs):
                    self.command, self.environment = command, kwargs['env']
                    self.pid, self.started, self.returncode = 1234, clock[0], None
                    self.terminated = False
                    processes.append(self)
                    run_name = command[command.index('--name') + 1]
                    for name in ['best.pt', 'last.pt']:
                        weight = root / 'experiments/runs' / run_name / 'weights' / name
                        weight.parent.mkdir(parents=True, exist_ok=True)
                        weight.touch()

                def poll(self):
                    if self.returncode is None and clock[0] - self.started >= 2:
                        self.returncode = 1 if fail else 0
                    return self.returncode

                def terminate(self):
                    self.returncode, self.terminated = -15, True

                def wait(self, timeout=None):
                    return self.returncode

            with patch.object(launcher, 'ROOT', root), patch.object(launcher, 'SOURCES', ['fixture.txt']), \
                 patch.object(launcher, 'csv_progress', return_value=None), \
                 patch.object(launcher.subprocess, 'Popen', Process), \
                 patch.object(launcher.time, 'sleep', lambda _: clock.__setitem__(0, clock[0] + 1)), \
                 patch('builtins.print'):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'train failed'):
                        launcher.run(50, 'test', gate, skip_test=True,
                                     yolo_weight=root / 'yolo.pt', detail_weight=root / 'detail.pt',
                                     archive=root / 'code.zip', data=root / 'data.yaml')
                else:
                    launcher.run(50, 'test', gate, skip_test=True,
                                 yolo_weight=root / 'yolo.pt', detail_weight=root / 'detail.pt',
                                 archive=root / 'code.zip', data=root / 'data.yaml')
            status = json.loads((root / 'experiments/analysis/geometry_v5_input64_multi_e50_test/status.json').read_text())
        return status, processes

    def test_one_process_three_shared_heads_no_alpha(self):
        state, processes = self.exercise()
        self.assertEqual(state['phase'], 'complete')
        self.assertEqual(len(processes), 1)
        self.assertTrue(state['one_joint_checkpoint'])
        cmd = processes[0].command
        self.assertIn('--multi-head-direct', cmd)
        self.assertIn('--corrected-geometry', cmd)
        self.assertIn('--geometry-input-support', cmd)
        self.assertNotIn('--corrected-direct-fusion', cmd)
        self.assertNotIn('--fusion-alpha', cmd)
        self.assertEqual(cmd[cmd.index('--epochs') + 1], '50')
        self.assertEqual(cmd[cmd.index('--lr0') + 1], '0.0001')
        self.assertEqual(cmd[cmd.index('--seed') + 1], '0')
        for flag, filename in [('--yolo-weight', 'yolo.pt'), ('--detail-weight', 'detail.pt'),
                               ('--archive', 'code.zip'), ('--data', 'data.yaml')]:
            self.assertEqual(Path(cmd[cmd.index(flag) + 1]).name, filename)
        self.assertEqual(state['initialization_seed'], 0)
        self.assertEqual(state['trainer_seed'], 0)
        self.assertTrue(state['deterministic'])
        self.assertEqual(state['scatter_patch_hw'], {'high': [8, 8], 'mid': [4, 4], 'low': [2, 2]})

    def test_failure_does_not_touch_other_jobs(self):
        state, processes = self.exercise(fail=True)
        self.assertEqual(state['phase'], 'failed')
        self.assertEqual(len(processes), 1)
        self.assertFalse(processes[0].terminated)  # already exited; nothing else killed


if __name__ == '__main__':
    unittest.main(verbosity=2)
