# Results Table

Official leaderboard artifacts supplied after competition completion.

| Artifact | 3D HOTA (%) | DetA (%) | AssA (%) | LocA (%) | Causality | Notes |
|---|---:|---:|---:|---:|---|---|
| `track1-yolo11x1920_dfine_p025_bev_gap60_cost115_v2.zip` | **12.4128** | 14.5213 | 11.8109 | 57.0038 | Offline | Best raw result, rank 9 |
| `track1-yolo11x1920_dfine_precision025_adaptive_online.zip` | 12.3778 | 14.5214 | 11.7634 | 57.0038 | Online | Strong causal detector-fusion baseline |
| `track1-yolo11x1920_resume_e29_dfine_p025_adaptive_online.zip` | 12.1237 | 14.5781 | 10.2219 | 57.0948 | Online | Higher DetA/LocA, weaker AssA |
| `track1-yolo11x1920_resume_e29_dfine_p025_mvopt_balanced.zip` | 12.1418 | 14.5837 | 10.2434 | 57.1022 | Verify | Depth/MV optimizer did not help enough |
| `track1-final_classhybrid_hota_depth_mvopt_fullclasses.zip` | 12.1249 | 14.5835 | 10.2271 | 57.1020 | Verify | Classwise hybrid did not beat best |
| `track1-yolo40e_dfine_precision025_bev_graph.zip` | 12.0640 | 13.9304 | 10.8072 | 57.0960 | Offline | Older detector base; lower DetA |

## Main Comparison

Best method vs strongest non-BEV baseline:

| Comparison | Delta 3D HOTA | Delta DetA | Delta AssA | Delta LocA |
|---|---:|---:|---:|---:|
| BEV gap60 cost1.15 vs adaptive online | +0.0350 | -0.0001 | +0.0475 | 0.0000 |

Interpretation:

The final BEV graph improves ID association slightly without changing localization. The official score gain comes almost entirely from AssA.
