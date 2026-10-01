### Table 4.1  Z-Score Threshold Sensitivity Sweep (W = 50, no ceiling)

| theta_z | TP | FP | FN | Precision | Recall | F1 | vs paper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.5 | 29 | 28558 | 16 | 0.0010 | 0.6444 | 0.0020 | DIFF (paper TP/FP/FN=(43, 34390, 2)) |
| 2.0 | 14 | 12296 | 31 | 0.0011 | 0.3111 | 0.0023 | DIFF (paper TP/FP/FN=(35, 12862, 10)) |
| 2.5 | 13 | 4170 | 32 | 0.0031 | 0.2889 | 0.0061 | DIFF (paper TP/FP/FN=(29, 2675, 16)) |
| 3.0 | 12 | 1122 | 33 | 0.0106 | 0.2667 | 0.0204 | DIFF (paper TP/FP/FN=(24, 263, 21)) |
| 3.5 | 12 | 245 | 33 | 0.0467 | 0.2667 | 0.0795 | DIFF (paper TP/FP/FN=(24, 13, 21)) |
| 4.0 | 12 | 59 | 33 | 0.1690 | 0.2667 | 0.2069 | DIFF (paper TP/FP/FN=(24, 0, 21)) |
| STATIC 80% | 39 | 25 | 6 | 0.6094 | 0.8667 | 0.7156 | OK |

### Table 4.2  Joint theta_z x theta_ceiling Sweep (W = 50)

| theta_z | theta_ceiling | TP | FP | FN | Precision | Recall | F1 | vs paper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.0 | 85% | 27 | 59 | 18 | 0.3140 | 0.6000 | 0.4122 | OK |
| 4.5 | 85% | 27 | 14 | 18 | 0.6585 | 0.6000 | 0.6279 | OK |
| 5.0 | 85% | 27 | 3 | 18 | 0.9000 | 0.6000 | 0.7200 | OK |
| 5.5 | 85% | 27 | 0 | 18 | 1.0000 | 0.6000 | 0.7500 | OK |
| 6.0 | 85% | 27 | 0 | 18 | 1.0000 | 0.6000 | 0.7500 | OK |
| 5.0 | 90% | 12 | 3 | 33 | 0.8000 | 0.2667 | 0.4000 | OK |
| 5.0 | none | 12 | 3 | 33 | 0.8000 | 0.2667 | 0.4000 | OK |
| STATIC | 80% | 39 | 25 | 6 | 0.6094 | 0.8667 | 0.7156 |  |

### Table 4.3  Final Comparative Performance

| Metric | Static Threshold (80%) | Enhanced (theta_z=5.5, ceiling=85%) | Delta | vs paper |
| --- | --- | --- | --- | --- |
| True Positives | 39 | 27 | -12 | OK / static OK |
| False Positives | 25 | 0 | -25 |  |
| False Negatives | 6 | 18 | +12 |  |
| Precision | 0.6094 | 1.0000 | +0.3906 |  |
| Recall | 0.8667 | 0.6000 | -0.2667 |  |
| F1-Score | 0.7156 | 0.7500 | +0.0344 |  |
