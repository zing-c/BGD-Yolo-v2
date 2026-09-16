# Direct input64 当前关键权重

| 文件 | 模型 | Val-best epoch | SHA256 |
|---|---|---:|---|
| [p3_best.pt](p3_best.pt) | P3 独立单头 | 45 | `03bbecbb8257ad933c8e381c8941e2b1f23dadcbad8f15ea5dcf172789357f7d` |
| [p4_best.pt](p4_best.pt) | P4 独立单头 | 9 | `ca1379869b5d1d5f99ab1d86bc895207a5227858d6e0be888e03e7ad0b412cff` |
| [p5_best.pt](p5_best.pt) | P5 独立单头 | 38 | `74af0b70567be09acd2473659427d843c1d7b063be39d5825efa94563d5fbaf0` |
| [multi_r2_best.pt](multi_r2_best.pt) | P3+P4+P5 联合三头 r2 | 40 | `aec33ba87c3f534740a4501b6a545fcf9d7e2c27f080207b7613e7b4b06bb2c0` |

best 均由 Val mAP50 选择，不用 Test 选 epoch。三个单头是三份独立联合权重；三头 r2 是共享 YOLO / Detail、包含三个独立融合投影的一份联合权重。这里只发布 best，不发布重复的 last。完整指标、参数和配置见[单头与三头总表](../../DIRECT_INPUT64_CURRENT_SINGLE_MULTI_RESULTS_20260916.md)。
