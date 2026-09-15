"""Repeat corrected Direct-head Test timing under one controlled environment.

Uses the existing checkpoint loader and model forward unchanged. Real-image
warmup is excluded from timing, CUDA is synchronized, and a read-only Detail
pre-hook records the actual crop count. No training or external logging occurs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from run_318_fusion_alpha import (
    DEFAULT_ARCHIVE,
    DEFAULT_DATA,
    ROOT,
    install_legacy_modules,
    load_joint_checkpoint,
    patch_corrected_single_head_direct,
    patch_multi_head_direct,
)


RUNS = {
    "p3": "best318_high_p3_detecthead_corrected_xy_scale8_sum_e50_lr1e4_r1",
    "p4": "best318_mid_p4_detecthead_correctedscatter_direct_e50_lr1e4_r1",
    "p5": "best318_low_p5_detecthead_corrected_xy_scale2_direct_e50_lr1e4_r2",
    "multi": "best318_multihead_p3p4p5_detecthead_corrected_xy_scaleaware_sum_e50_lr1e4_r3_w8",
}
REFERENCES = {
    "p3": [0.9419642857142857, 0.9017094017094017, 0.9320060811996151,
           0.9058516219903274, 0.8777653678383533],
    "p4": [0.9631225106711019, 0.892883718042674, 0.9381313772500867,
           0.9105921324398474, 0.8845447318353198],
    "p5": [0.9502262443438914, 0.8974358974358975, 0.9344706981330392,
           0.9163943373026056, 0.8846609109529167],
    "multi": [0.9551569506726457, 0.9102564102564102, 0.9426511477765371,
              0.9214399201034569, 0.8896087383880467],
}
METRIC_NAMES = ("precision", "recall", "map50", "map75", "map50_95")


def gpu_snapshot() -> dict:
    output = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    apps = [
        {"pid": int(row[0]), "name": row[1].strip(), "memory_mib": row[2].strip()}
        for row in csv.reader(output.splitlines()) if row
    ]
    foreign = [
        app for app in apps
        if app["pid"] != os.getpid() and not any(
            token in app["name"].lower()
            for token in ("anydesk", "gnome-remote-desktop", "xorg", "gnome-shell")
        )
    ]
    state = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,clocks.sm",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    snapshot = {"at": datetime.now().isoformat(), "apps": apps, "gpu": state}
    if foreign:
        raise RuntimeError(f"Other GPU compute process appeared: {foreign}")
    return snapshot


def validator_class(warmup_images: int, snapshots: list):
    from ultralytics.yolo.v8.detect.val import DetectionValidator

    class TimingValidator(DetectionValidator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.samples = []
            self._crop_count = 0
            self._backend = None
            self._forward = None
            self._crop_handle = None
            self.callbacks["on_val_batch_end"].append(self.monitor_gpu)

        def monitor_gpu(self, _validator):
            if (self.batch_i + 1) % 400 == 0:
                snapshots.append(gpu_snapshot())
                print("BENCHMARK_PROGRESS " + json.dumps({
                    "image": self.batch_i + 1,
                    "mean_forward_ms": statistics.mean(s["forward_ms"] for s in self.samples),
                    "total_detail_crops": sum(s["detail_crops"] for s in self.samples),
                }), flush=True)

        def init_metrics(self, model):
            super().init_metrics(model)
            self._backend = model
            inner = model.model
            if next(inner.parameters()).dtype != torch.float32:
                raise RuntimeError("Benchmark requires FP32 for all models")
            # Read dataset items directly: do not consume or rotate the
            # InfiniteDataLoader iterator used for the timed full Test pass.
            started = time.perf_counter()
            for index in range(min(warmup_images, len(self.dataloader.dataset))):
                dataset = self.dataloader.dataset
                batch = dataset.collate_fn([dataset[index]])
                batch = super().preprocess(batch)
                model.batch = batch
                inner.batch = batch
                model(batch["img"], augment=False)
            torch.cuda.synchronize(self.device)
            model.batch = None
            inner.batch = None
            print("BENCHMARK_WARMUP_DONE " + json.dumps({
                "images": min(warmup_images, len(self.dataloader.dataset)),
                "seconds": time.perf_counter() - started,
            }), flush=True)
            snapshots.append(gpu_snapshot())

            def count_crops(_module, inputs):
                self._crop_count += int(inputs[0].shape[0])

            self._crop_handle = inner.detail_model.register_forward_pre_hook(count_crops)
            self._forward = model.forward

            def measured_forward(images, *args, **kwargs):
                self._crop_count = 0
                torch.cuda.synchronize(self.device)
                started = time.perf_counter()
                result = self._forward(images, *args, **kwargs)
                torch.cuda.synchronize(self.device)
                self.samples.append({
                    "image_index": len(self.samples),
                    "image": str(model.batch["im_file"][0]),
                    "input_shape": list(images.shape),
                    "forward_ms": (time.perf_counter() - started) * 1000,
                    "detail_crops": self._crop_count,
                })
                return result

            model.forward = measured_forward

        def close(self):
            if self._crop_handle is not None:
                self._crop_handle.remove()
            if self._backend is not None and self._forward is not None:
                self._backend.forward = self._forward

    return TimingValidator


def write_reports(output: Path, protocol: dict, trials: list, status: str) -> None:
    summary = {}
    for name in protocol["models"]:
        passes = [trial for trial in trials if trial["model"] == name]
        if not passes:
            continue
        inference = [trial["speed_ms_per_image"]["inference"] for trial in passes]
        summary[name] = {
            "passes": len(passes),
            "inference_ms_mean": statistics.mean(inference),
            "inference_ms_std_between_passes": statistics.stdev(inference) if len(passes) > 1 else None,
            "inference_ms_each_pass": inference,
            "detail_crops_per_image": statistics.mean(t["detail_crops_per_image"] for t in passes),
            "accuracy_each_pass": [t["metrics"] for t in passes],
            "reference_matches": [t["reference_match"] for t in passes],
        }
    with (output / "results.json").open("w", encoding="utf-8") as stream:
        json.dump({"status": status, "protocol": protocol, "summary": summary,
                   "trials": trials}, stream, indent=2, ensure_ascii=False)
    with (output / "summary.md").open("w", encoding="utf-8") as stream:
        stream.write("# Corrected Direct heads: controlled Test timing\n\n")
        stream.write(f"Status: {status}\n\n")
        stream.write(f"CUDA:0, imgsz=640 (rectangular letterbox), batch=1, FP32, workers={protocol['workers']}, "
                     "candidate conf=0.2, final conf=0.5, NMS IoU=0.5. "
                     "Real-image warmup excluded; synchronized CUDA timing; plots disabled.\n\n")
        stream.write("| Model | Passes | Inference ms/image: mean ± sample std across passes | "
                     "Detail crops/image |\n|---|---:|---:|---:|\n")
        for name, row in summary.items():
            std = row["inference_ms_std_between_passes"]
            spread = "pending" if std is None else f"{std:.3f}"
            stream.write(f"| {name} | {row['passes']} | {row['inference_ms_mean']:.3f} ± "
                         f"{spread} | {row['detail_crops_per_image']:.3f} |\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--models", nargs="+", choices=RUNS, default=list(RUNS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-images", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "analysis" /
                        f"corrected_direct_timing_{datetime.now():%Y%m%d_%H%M%S}")
    args = parser.parse_args()
    if args.rounds < 1 or args.warmup_images < 1:
        parser.error("rounds and warmup-images must be positive")
    os.environ["WANDB_MODE"] = "disabled"
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    trials = []
    protocol = {
        "started_at": datetime.now().isoformat(), "models": args.models,
        "rounds": args.rounds, "warmup_images": args.warmup_images,
        "workers": args.workers, "torch_cpu_threads": args.threads,
        "precision": "FP32", "device": "cuda:0", "imgsz": 640, "rect": True,
        "batch": 1, "candidate_conf": 0.2, "conf": 0.5, "iou": 0.5,
        "plots": False, "data": str(args.data.resolve()),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "initial_gpu": gpu_snapshot(),
    }
    print("BENCHMARK_PROTOCOL " + json.dumps(protocol), flush=True)
    write_reports(args.output, protocol, trials, "running")

    with tempfile.TemporaryDirectory(prefix="bgd_controlled_timing_") as temporary:
        legacy = install_legacy_modules(DEFAULT_ARCHIVE, temporary, use_fusion_alpha=False)
        original_predict = legacy.BGD_YOLO._predict_once
        original_set_hook = legacy.BGD_YOLO.set_hook
        from ultralytics.yolo.cfg import get_cfg
        from ultralytics.yolo.utils import DEFAULT_CFG, callbacks
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        for round_index in range(args.rounds):
            order = args.models[round_index % len(args.models):] + args.models[:round_index % len(args.models)]
            for name in order:
                snapshots = [gpu_snapshot()]
                checkpoint = ROOT / "experiments" / "runs" / RUNS[name] / "weights" / "best.pt"
                legacy.BGD_YOLO._predict_once = original_predict
                legacy.BGD_YOLO.set_hook = original_set_hook
                if name == "multi":
                    patch_multi_head_direct(legacy)
                else:
                    patch_corrected_single_head_direct(legacy)
                joint = load_joint_checkpoint(SimpleNamespace(checkpoint=checkpoint))
                config = joint.model.bgd_318_alpha_config
                if config["candidate_conf"] != 0.2 or config.get("scatter_reduce", "sum") != "sum":
                    raise RuntimeError(f"Incorrect checkpoint configuration: {config}")
                if name == "multi" and not config.get("multi_head_scale_aware"):
                    raise RuntimeError("Multi-head checkpoint is not corrected scale-aware")
                overrides = joint.overrides.copy()
                overrides.update({
                    "mode": "val", "data": str(args.data.resolve()), "split": "test",
                    "imgsz": 640, "rect": True, "batch": 1, "half": False,
                    "device": "0", "workers": args.workers, "conf": 0.5, "iou": 0.5,
                    "augment": False, "plots": False, "save_json": False, "save_txt": False,
                    "project": str(args.output), "name": f"round{round_index + 1}_{name}",
                })
                Validator = validator_class(args.warmup_images, snapshots)
                validator = Validator(args=get_cfg(DEFAULT_CFG, overrides=overrides),
                                      _callbacks=callbacks.get_default_callbacks())
                print("BENCHMARK_TRIAL_START " + json.dumps({
                    "round": round_index + 1, "model": name, "checkpoint": str(checkpoint),
                    "method": config["method"],
                }), flush=True)
                started = time.perf_counter()
                try:
                    validator(model=joint.model)
                finally:
                    validator.close()
                samples = validator.samples
                if len(samples) != validator.seen:
                    raise RuntimeError("Per-image sample count differs from evaluated images")
                metrics = validator.metrics
                accuracy = [float(metrics.box.mp), float(metrics.box.mr), float(metrics.box.map50),
                            float(metrics.box.map75), float(metrics.box.map)]
                trial = {
                    "round": round_index + 1, "model": name, "checkpoint": str(checkpoint),
                    "method": config["method"], "images": validator.seen,
                    "metrics": dict(zip(METRIC_NAMES, accuracy)),
                    "reference_match": bool(np.allclose(accuracy, REFERENCES[name], atol=1e-5, rtol=0))
                    if validator.seen == 1598 else None,
                    "speed_ms_per_image": dict(metrics.speed),
                    "measured_forward_ms_mean": statistics.mean(s["forward_ms"] for s in samples),
                    "measured_forward_ms_p50": float(np.percentile([s["forward_ms"] for s in samples], 50)),
                    "measured_forward_ms_p95": float(np.percentile([s["forward_ms"] for s in samples], 95)),
                    "detail_crops_total": sum(s["detail_crops"] for s in samples),
                    "detail_crops_per_image": sum(s["detail_crops"] for s in samples) / len(samples),
                    "validation_wall_seconds_including_warmup": time.perf_counter() - started,
                    "gpu_snapshots": snapshots + [gpu_snapshot()],
                }
                with (args.output / f"round{round_index + 1}_{name}_images.json").open("w") as stream:
                    json.dump(samples, stream)
                trials.append(trial)
                write_reports(args.output, protocol, trials, "running")
                print("BENCHMARK_TRIAL_DONE " + json.dumps(trial), flush=True)
                del validator, joint, metrics, samples
                gc.collect()
                torch.cuda.empty_cache()
    protocol["finished_at"] = datetime.now().isoformat()
    write_reports(args.output, protocol, trials, "complete")
    print("BENCHMARK_COMPLETE " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
