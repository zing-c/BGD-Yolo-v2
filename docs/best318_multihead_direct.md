# Best318 multi-head direct Grad-CAM fusion

This experiment extends the audited best318-era single-head path to the three
YOLO Detect classification levels simultaneously.

## Method

- Hook `Detect.cv3[0][1]`, `Detect.cv3[1][1]`, and `Detect.cv3[2][1]` in stable
  P3/P4/P5 order.
- Run one target backward pass and aggregate the three scale-normalized CAMs.
- Locate and crop candidate detail regions once, then run the Detail encoder
  once.
- Scatter the Detail features independently at each Detect feature-map scale.
- Use an independent `Conv2d(67,64,3) + BatchNorm2d + ReLU` projection per
  scale.
- Directly replace the classification logits at every selected level.

There is no alpha interpolation or `fusion_alpha` parameter in this mode. The
checkpoint method identifier is `best318_framework_multi_head_direct_v1`.

## Training run

- heads: high/P3, mid/P4, low/P5
- epochs: 50
- optimizer: AdamW
- initial LR: 0.0001
- batch: 32
- workers: 2
- AMP: enabled
- candidate confidence: 0.2
- W&B: [dtj4tsb4](https://wandb.ai/zing_c-ningbo-university/BGD-YOLO/runs/dtj4tsb4)

The one-image full forward smoke test completed successfully before the formal
run. The formal run also completed its first optimizer batches, confirming the
multi-head path is differentiable in the trainer.
