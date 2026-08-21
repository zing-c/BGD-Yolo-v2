# Active Detail architecture

The active project Detail model is `zip_compressed_d12_v1`.

- ZIP compressed route widths: large `8/16/32/32`, medium `8/16/32`, small `8/16`.
- Residual activation order: `ReLU(F(x) + shortcut)`.
- Route output size for a 64x64 crop: 4x4.
- Spatial attention dilations: 1 and 2. Dilation 5 and 7 were removed because
  they reduce to center-only sampling on a 4x4 feature map.
- V1 feature output remains 3x4x4, so the existing gated residual YOLO fusion
  is unchanged.
- V2/V3 semantic output remains configurable (64 channels by default).
- Required pretrained checkpoint:
  `run/detail_net_atten/zip_compressed_d12_v1.pt`.
- Checkpoint SHA-256:
  `88b78aba51e89b07c6cefde4333d498e5cc34523ab871601323b2980c37b48ad`.
- Best standalone validation (epoch 12/40): accuracy 91.23%, precision
  89.50%, recall 96.06%, F1 92.66%.

Historical wide `exp3_4_1.pt` weights are intentionally rejected by the new
loader. The archived wide implementation is documented under `archives/`.
