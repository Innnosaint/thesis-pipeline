"""
paper_check.py
Reproduces, from the pipeline's own output files, every number in Chapter Four
that the sweep scripts do not print: dataset description (4.1), ground-truth
counts (4.2), static-80% error analysis (4.3), the absorption diagnostics (4.4),
Table 4.4, the priority breakdown (4.7) and Table 4.5 (bootstrap CIs).

Reads  : final_scored_stream.jsonl  (run_pipeline.py, theta_z=5.5, ceiling=85, W=50)
         labels.jsonl               (build_ground_truth.py)
Usage  : python paper_check.py [--final PATH] [--labels PATH] [--seed 42] [--resamples 5000]
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

COLS = ["TP", "FP", "FN", "TN", "Precision", "Recall", "F1"]


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def confusion(pred, truth):
    tp = sum(p and t for p, t in zip(pred, truth))
    fp = sum(p and not t for p, t in zip(pred, truth))
    fn = sum((not p) and t for p, t in zip(pred, truth))
    tn = len(pred) - tp - fp - fn
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, Precision=P, Recall=R, F1=F)


def pct(v, q):
    s = sorted(v)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="../data/processed/final_scored_stream.jsonl")
    ap.add_argument("--labels", default="../data/ground_truth/labels.jsonl")
    ap.add_argument("--out-dir", default="../results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resamples", type=int, default=5000)
    a = ap.parse_args()

    rows = load(a.final)
    lab = {r["timestamp"]: r for r in load(a.labels)}
    rows = [r for r in rows if r["timestamp"] in lab]
    for r in rows:
        r["_y"] = bool(lab[r["timestamp"]]["is_anomaly"])
        r["_type"] = lab[r["timestamp"]].get("anomaly_type")
    x = [r["disk_usage_value"] for r in rows]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 4.1
    ready = [r for r in rows if r.get("baseline_ready")]
    print("=== 4.1 Dataset ===")
    print(f"readings: {len(rows):,}   first: {rows[0]['timestamp']}   last: {rows[-1]['timestamp']}")
    print(f"median {statistics.median(x):.1f}%   95th pct {pct(x, .95):.1f}%   range {min(x):.1f}-{max(x):.1f}%")
    print(f"warm-up excluded: {len(rows) - len(ready)}   evaluated: {len(ready):,}")
    print("paper: 105,122 readings; median 45.0; p95 54.6; range 26.6-100.2; 105,112 evaluated")

    # ---------------------------------------------------------------- 4.2
    pos_all = [r for r in rows if r["_y"]]
    thr = [r for r in pos_all if r["_type"] == "threshold"]
    kw = [r for r in pos_all if str(r["_type"]).startswith("keyword")]
    print("\n=== 4.2 Ground truth (all readings) ===")
    print(f"anomalies: {len(pos_all)} ({100 * len(pos_all) / len(rows):.3f}%)   by threshold: {len(thr)}   by keyword: {len(kw)}")
    if kw:
        print(f"keyword types: {sorted(set(r['_type'] for r in kw))}   usage {min(r['disk_usage_value'] for r in kw):.0f}-{max(r['disk_usage_value'] for r in kw):.0f}%")
    print("paper: 45 anomalies (0.043%): 39 threshold + 6 keyword ('disk quota', 38-55%)")

    # ---------------------------------------------------------------- 4.3 static
    y = [r["_y"] for r in ready]
    s80 = [r["disk_usage_value"] >= 80 for r in ready]
    s85 = [r["disk_usage_value"] >= 85 for r in ready]
    c80 = confusion(s80, y)
    fp80 = [r["disk_usage_value"] for r, p in zip(ready, s80) if p and not r["_y"]]
    fn80 = [r["disk_usage_value"] for r, p in zip(ready, s80) if (not p) and r["_y"]]
    print("\n=== 4.3 Static 80% ===")
    print(f"TP {c80['TP']}  FP {c80['FP']}  FN {c80['FN']}  P {c80['Precision']:.4f}  R {c80['Recall']:.4f}")
    if fp80: print(f"false positives span {min(fp80):.1f}-{max(fp80):.1f}%")
    if fn80: print(f"false negatives span {min(fn80):.1f}-{max(fn80):.1f}%")
    print("paper: TP 39, FP 25 (80.5-84.7%), FN 6 (38.3-54.9%), P 0.6094, R 0.8667")

    # ---------------------------------------------------------------- Table 4.4
    abn = [r.get("classification") == "ABNORMAL" for r in ready]
    sup = [bool(r.get("suppressed")) for r in ready]
    methods = {
        "Static 80%": s80, "Static 85%": s85,
        "Enhanced (suppression on)": [p and not s for p, s in zip(abn, sup)],
        "Hybrid (priority only)": abn,
    }
    res = {k: confusion(v, y) for k, v in methods.items()}
    base = res["Static 80%"]["F1"]
    print("\n=== Table 4.4 ===")
    hdr = ["Metric"] + list(methods)
    table = []
    for m in COLS:
        table.append([m] + [res[k][m] for k in methods])
    table.append(["F1 vs Static 80%"] + ["n/a" if k == "Static 80%" else f"{res[k]['F1'] - base:+.4f}" for k in methods])
    fmt = lambda v: f"{v:,}" if isinstance(v, int) else f"{v:.4f}" if isinstance(v, float) else str(v)
    w = [max(len(str(h)), *(len(fmt(r[i])) for r in table)) for i, h in enumerate(hdr)]
    print("  ".join(h.rjust(wd) for h, wd in zip(hdr, w)))
    for r in table:
        print("  ".join(fmt(v).rjust(wd) for v, wd in zip(r, w)))
    with open(out / "table_4_4.csv", "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(hdr); cw.writerows(table)
    print("paper: Static80 39/25/6  Static85 39/0/6 F1 0.9286  Enhanced 27/0/18 F1 0.7500 (+0.0344)  Hybrid 39/0/6 F1 0.9286")

    # ---------------------------------------------------------------- 4.6 / 4.7 correlation breakdown
    flagged = [r for r in ready if r.get("classification") == "ABNORMAL"]
    cnt = {p: sum(r.get("priority") == p for r in flagged) for p in ("HIGH", "MEDIUM", "LOW")}
    print("\n=== 4.6-4.7 Correlation breakdown ===")
    print(f"flagged before correlation: {len(flagged)} (true anomalies among them: {sum(r['_y'] for r in flagged)})")
    print(f"HIGH {cnt['HIGH']}   MEDIUM {cnt['MEDIUM']}   LOW (suppressed) {cnt['LOW']}   delivered alerts: {cnt['HIGH'] + cnt['MEDIUM']}")
    alerts = [r for r in flagged if not r.get("suppressed")]
    alerts.sort(key=lambda r: r["enhanced_alert_score"], reverse=True)
    k = cnt["HIGH"]
    print(f"top-{k} ranked alerts all HIGH: {all(r.get('priority') == 'HIGH' for r in alerts[:k])}")
    print("paper: 39 flagged -> HIGH 12, LOW 12 (suppressed), MEDIUM 15; 27 delivered; HIGH at ranks 1-12")
    print("top 5 alerts:")
    for i, r in enumerate(alerts[:5], 1):
        print(f"  {i}. [{r['priority']}] score={r['enhanced_alert_score']:.2f} z={r['zscore']:.2f} disk={r['disk_usage_value']}% {r['timestamp']} kw={r.get('matched_keyword')}")

    # ---------------------------------------------------------------- 4.4 diagnostics
    print("\n=== 4.4 Diagnostics ===")
    sig_all = statistics.median(r["baseline_sigma"] for r in ready)
    print(f"median baseline sigma (all readings): {sig_all:.2f}   (paper: 3.00)")
    day = [r for r in ready if r["timestamp"].startswith("2024-02-02") and r["disk_usage_value"] >= 85]
    if day:
        zs = [abs(r["zscore"]) for r in day]
        print(f"2024-02-02 readings >=85%: n={len(day)} usage {min(r['disk_usage_value'] for r in day):.1f}-{max(r['disk_usage_value'] for r in day):.1f}%"
              f"  |Z| {min(zs):.2f}-{max(zs):.2f}  mu {min(r['baseline_mu'] for r in day):.1f}-{max(r['baseline_mu'] for r in day):.1f}"
              f"  sigma up to {max(r['baseline_sigma'] for r in day):.1f}")
        print("paper: 85.2-89.3%, |Z| 1.66-2.03, mu ~68-70, sigma ~9.7")
    big = sorted((r for r in flagged if r.get("suppressed")), key=lambda r: -abs(r["zscore"]))[:3]
    for r in big:
        print(f"suppressed-as-benign example: disk={r['disk_usage_value']}% z={r['zscore']:.1f} kw={r.get('matched_keyword')} @ {r['timestamp']}")
    print("paper: a 97.7% reading with Z = 13.3 was discarded by correlation as benign")

    # ---------------------------------------------------------------- Table 4.5
    B = a.resamples
    yy = np.array(y); ss = np.array(methods["Static 80%"]); ee = np.array(methods["Enhanced (suppression on)"])
    cells = yy.astype(int) * 4 + ss.astype(int) * 2 + ee.astype(int)
    counts = np.bincount(cells, minlength=8)
    N = int(counts.sum())
    draws = np.random.default_rng(a.seed).multinomial(N, counts / N, size=B)
    cell = np.arange(8); cy, cs, ce = (cell >> 2) & 1, (cell >> 1) & 1, cell & 1

    def prf(bit):
        tp = draws[:, (cy == 1) & (bit == 1)].sum(1); fp = draws[:, (cy == 0) & (bit == 1)].sum(1)
        fn = draws[:, (cy == 1) & (bit == 0)].sum(1)
        div = lambda p, q: np.divide(p, q, out=np.zeros(B), where=q > 0)
        return div(tp, tp + fp), div(tp, tp + fn), div(2 * tp, 2 * tp + fp + fn)

    sp, sr, sf = prf(cs); ep, er, ef = prf(ce)
    ci = lambda v: f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"
    d = ef - sf
    print(f"\n=== Table 4.5 Bootstrap 95% CIs ({B:,} resamples, seed={a.seed}) ===")
    print(f"{'Metric':<10}{'Static 80%':<20}{'Enhanced (suppression on)'}")
    t45 = [("Precision", ci(sp), ci(ep)), ("Recall", ci(sr), ci(er)), ("F1", ci(sf), ci(ef))]
    for m, s_, e_ in t45:
        print(f"{m:<10}{s_:<20}{e_}")
    print(f"paired F1 difference: mean {d.mean():+.3f}  95% CI {ci(d)}  Enhanced>Static in {100 * (d > 0).mean():.1f}% of resamples")
    tp_e = res["Enhanced (suppression on)"]["TP"]
    if tp_e:
        print(f"rule of three: upper bound on false-alert rate ~ 3/{tp_e} = {3 / tp_e:.2f}")
    print("paper: P [0.484,0.732]/[1,1]  R [0.756,0.958]/[0.447,0.744]  F1 [0.608,0.808]/[0.618,0.853]  diff +0.033 [-0.095,0.164] 69.2%  3/27=0.11")
    print("(bootstrap endpoints agree to about +/-0.01 unless you use the exact seed/script behind the paper)")
    with open(out / "table_4_5.csv", "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["Metric", "Static 80%", "Enhanced (suppression on)"]); cw.writerows(t45)
    print(f"\nSaved: {out / 'table_4_4.csv'}, {out / 'table_4_5.csv'}")


if __name__ == "__main__":
    main()
