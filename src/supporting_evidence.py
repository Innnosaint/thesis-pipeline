"""
supporting_evidence.py
Produces the evidence behind the claims made in the thesis text (not just the
headline tables):

  A. Dataset facts            (Sec. 4.1: reading count, span, median / 95th pct)
  B. Ground-truth breakdown   (Sec. 4.2: threshold vs keyword labels)
  C. Objective 1              (Sec. 4.3: static-threshold FPs / FNs inspected)
  D. Recall-gap diagnosis     (Sec. 4.4: FNs at theta_z=4.0 are gradual climbs
                               whose Z-score peaks ~1.6-2.0)
  E. Residual FNs            (Sec. 4.6: why the final config still misses 18)
  F. Correlation audit        (Sec. 4.8: suppressed / HIGH counts, ranking check)
  G. Statistical confidence   (bootstrap 95% CIs, paired F1 difference)
  H. Runtime                  (Problem 4: +/-dt window vs full-corpus matching)

Usage (from src/):
  python supporting_evidence.py | tee ../results/evidence_report.txt
"""
import argparse
import random
import statistics
import sys
import time
from bisect import bisect_left
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from baseline import run_baseline_over_stream
from zscore_detector import classify
from log_correlation import (correlate_and_score, get_log_window,
                             classify_priority, build_time_index)
from alert_scoring import rank_alerts
from common import load_stream, load_context, make_cfg

ROOT = Path(__file__).resolve().parent.parent


def title(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def pct(sorted_vals, q):
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def key(e):
    return (e["log_source"], e["timestamp"].isoformat())


def read_labels_full(path):
    import json
    d = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                d[(r["log_source"], r["timestamp"])] = r
    return d


def ctx_priority(ctx, ctx_ts, ts, cfg):
    win = get_log_window(ctx, ctx_ts, ts, cfg["correlation"]["delta_t_seconds"])
    return classify_priority(win, cfg["correlation"]["benign_keywords"],
                             cfg["correlation"]["failure_keywords"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clean-stream", default=str(ROOT / "data/processed/clean_stream.jsonl"))
    ap.add_argument("--labels", default=str(ROOT / "data/ground_truth/labels.jsonl"))
    ap.add_argument("--context", default=None)
    ap.add_argument("--window-size", type=int, default=50)
    ap.add_argument("--static-threshold", type=float, default=80.0)
    ap.add_argument("--theta-final", type=float, default=5.5)
    ap.add_argument("--ceiling-final", type=float, default=85.0)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    raw = load_stream(a.clean_stream)
    lab_full = read_labels_full(a.labels)
    labels = {k: bool(v["is_anomaly"]) for k, v in lab_full.items()}
    context = load_context(a.clean_stream, a.context)
    if context:
        context.sort(key=lambda e: e["timestamp"])
    cfg = make_cfg(60)
    base = run_baseline_over_stream([dict(e) for e in raw], window_size=a.window_size,
                                    min_observations=max(5, int(a.window_size * 0.2)))
    ctx = context if context else sorted(base, key=lambda e: e["timestamp"])
    ctx_ts = [c["timestamp"] for c in ctx]

    # ------------------------------------------------------------------ A
    title("A. DATASET FACTS (Section 4.1)")
    vals = sorted(e["disk_usage_value"] for e in raw)
    ts = [e["timestamp"] for e in raw]
    gaps = sorted((ts[i + 1] - ts[i]).total_seconds() for i in range(len(ts) - 1))
    span = (ts[-1] - ts[0]).total_seconds() / 86400
    print(f"Readings                : {len(raw)}")
    print(f"First / last            : {ts[0]} -> {ts[-1]}")
    print(f"Span                    : {span:.1f} days   (paper says 260 in Ch.4, 263 in Ch.5)")
    print(f"Median sampling gap     : {statistics.median(gaps) / 60:.1f} min")
    print(f"Implied span @ this many readings & 5-min sampling: {len(raw) * 5 / 1440:.1f} days")
    print(f"Disk usage median / p95 : {statistics.median(vals):.1f}% / {pct(vals, .95):.1f}%   (paper: 45.0% / 54.6%)")
    print(f"Min / max               : {vals[0]:.1f}% / {vals[-1]:.1f}%")

    # ------------------------------------------------------------------ B
    title("B. GROUND-TRUTH BREAKDOWN (Section 4.2)")
    n_true = sum(labels.values())
    by_type = {}
    for r in lab_full.values():
        if r["is_anomaly"]:
            t = r.get("anomaly_type") or "unknown"
            t = "keyword" if str(t).startswith("keyword") else t
            by_type[t] = by_type.get(t, 0) + 1
    print(f"Labeled readings        : {len(labels)}")
    print(f"True anomalies          : {n_true} ({n_true / len(labels) * 100:.3f}%)   (paper: 45, 0.04%)")
    for t, n in sorted(by_type.items()):
        print(f"  by {t:<18}: {n}")
    print("  (paper: 39 by threshold + 6 by keyword)")

    # ------------------------------------------------------------------ C
    title("C. OBJECTIVE 1 -- STATIC 80% THRESHOLD ERRORS (Section 4.3)")
    fps, fns = [], []
    for e in base:
        if not e["baseline_ready"] or key(e) not in labels:
            continue
        pred = e["disk_usage_value"] >= a.static_threshold
        if pred and not labels[key(e)]:
            fps.append(e)
        elif (not pred) and labels[key(e)]:
            fns.append(e)
    n_fp_benign = n_fp_fail = n_fp_none = 0
    for e in fps:
        pr, kw = ctx_priority(ctx, ctx_ts, e["timestamp"], cfg)
        n_fp_benign += pr == "LOW"; n_fp_fail += pr == "HIGH"; n_fp_none += pr == "MEDIUM"
    print(f"Static false positives  : {len(fps)}  (paper: 25)")
    print(f"  with a benign keyword in +/-60s  : {n_fp_benign}")
    print(f"  with a failure keyword in +/-60s : {n_fp_fail}   (paper claims none)")
    print(f"  with no keyword either way       : {n_fp_none}")
    if fps:
        v = [e["disk_usage_value"] for e in fps]
        print(f"  FP usage range                   : {min(v):.1f}% - {max(v):.1f}%")
    print(f"Static false negatives  : {len(fns)}  (paper: 6)")
    for e in fns[:10]:
        pr, kw = ctx_priority(ctx, ctx_ts, e["timestamp"], cfg)
        print(f"  {e['timestamp']}  {e['disk_usage_value']:.1f}%  nearby keyword: {kw}")

    # ------------------------------------------------------------------ D
    title("D. RECALL-GAP DIAGNOSIS at theta_z=4.0, no ceiling (Section 4.4)")
    cls = classify([dict(e) for e in base], theta_z=4.0, hard_ceiling=None)
    scored = correlate_and_score(cls, ctx, cfg)
    all_ts = [e["timestamp"] for e in scored]
    pos = {id(e): i for i, e in enumerate(scored)}
    fn4 = [e for e in scored if e["baseline_ready"] and labels.get(key(e))
           and not (e["classification"] == "ABNORMAL" and not e["suppressed"])]
    print(f"False negatives at theta_z=4.0: {len(fn4)}  (paper: 21)")
    gradual, zs = [], []
    for e in fn4:
        lo = bisect_left(all_ts, e["timestamp"] - timedelta(hours=12))
        back = [x["disk_usage_value"] for x in scored[lo:pos[id(e)] + 1]]
        start_min = min(back) if back else e["disk_usage_value"]
        rise = e["disk_usage_value"] - start_min
        hours = (e["timestamp"] - scored[lo]["timestamp"]).total_seconds() / 3600
        # heuristic for "gradual climb": rose >= 8 points over the prior <=12h
        # starting from a 60-80% level, and the Z-score stayed below theta
        is_grad = rise >= 8 and 60 <= start_min <= 80 and abs(e["zscore"]) < 4.0
        if is_grad:
            gradual.append((e, start_min, rise)); zs.append(abs(e["zscore"]))
    print(f"Matching 'gradual climb' heuristic (rise>=8 pts in <=12h from 60-80%): "
          f"{len(gradual)} / {len(fn4)}   (paper: 19 / 21)")
    if zs:
        print(f"  |Z| at those readings : min {min(zs):.2f}  median {statistics.median(zs):.2f}  "
              f"max {max(zs):.2f}   (paper: ~1.6 - 2.0)")
        sv = [g[1] for g in gradual]; pv = [g[0]["disk_usage_value"] for g in gradual]
        print(f"  climb start / peak    : {min(sv):.0f}-{max(sv):.0f}%  ->  {min(pv):.0f}-{max(pv):.0f}%   "
              f"(paper: 66-76% -> 85-89%)")
        absorbed = [g[0]["baseline_mu"] for g in gradual]
        print(f"  baseline mu at peak   : {min(absorbed):.1f} - {max(absorbed):.1f}% "
              f"(mu climbing with the anomaly = baseline absorption)")
    print("\nFirst 8 false negatives (timestamp, value, mu, sigma, Z):")
    for e in fn4[:8]:
        print(f"  {e['timestamp']}  x={e['disk_usage_value']:.1f}  mu={e['baseline_mu']:.1f}  "
              f"sigma={e['baseline_sigma']:.2f}  Z={e['zscore']:.2f}")

    # ------------------------------------------------------------------ E / F
    title("E. RESIDUAL ERRORS OF THE FINAL CONFIGURATION (Section 4.6)")
    clsf = classify([dict(e) for e in base], theta_z=a.theta_final, hard_ceiling=a.ceiling_final)
    fin = correlate_and_score(clsf, ctx, cfg)
    fnf = [e for e in fin if e["baseline_ready"] and labels.get(key(e))
           and not (e["classification"] == "ABNORMAL" and not e["suppressed"])]
    reasons = {"suppressed as benign by correlation": 0, "usage 80-84.9%, never flagged": 0,
               "usage <80%, never flagged": 0, "usage >=85% but not flagged": 0}
    for e in fnf:
        v = e["disk_usage_value"]
        if e["classification"] == "ABNORMAL" and e["suppressed"]:
            reasons["suppressed as benign by correlation"] += 1
        elif v >= a.ceiling_final:
            reasons["usage >=85% but not flagged"] += 1
        elif v >= 80:
            reasons["usage 80-84.9%, never flagged"] += 1
        else:
            reasons["usage <80%, never flagged"] += 1
    print(f"False negatives at final config: {len(fnf)}  (paper: 18)")
    for r, n in reasons.items():
        print(f"  {r:<42}: {n}")
    print("  (paper attributes the misses to gradual climbs plateauing at 80-84%)")

    title("F. CONTEXTUAL CORRELATION AUDIT (Section 4.8)")
    flagged = [e for e in fin if e["classification"] == "ABNORMAL" and e["baseline_ready"]]
    def cnt(pred): return sum(1 for e in flagged if pred(e))
    tp_flag = [e for e in flagged if labels.get(key(e))]
    print(f"ABNORMAL before suppression      : {len(flagged)}")
    print(f"  of which true anomalies        : {len(tp_flag)}")
    print(f"  priority HIGH / MEDIUM / LOW   : {cnt(lambda e: e['priority']=='HIGH')} / "
          f"{cnt(lambda e: e['priority']=='MEDIUM')} / {cnt(lambda e: e['priority']=='LOW')}")
    print(f"Delivered alerts (non-suppressed): {sum(1 for e in flagged if not e['suppressed'])}")
    sup_true = sum(1 for e in tp_flag if e["suppressed"])
    high_true = sum(1 for e in tp_flag if e["priority"] == "HIGH")
    print(f"True anomalies SUPPRESSED        : {sup_true}   <-- these count as FALSE NEGATIVES in all tables")
    print(f"True anomalies escalated HIGH    : {high_true}")
    print("  Paper Sec. 4.8 says 12 suppressed + 12 HIGH 'among the 27 correctly detected true positives'.")
    print("  Suppressed alerts are not delivered, so they cannot also be true positives -- reconcile before defending.")
    ranked = rank_alerts([dict(e) for e in fin if e["baseline_ready"]])
    first_non_high = next((i for i, e in enumerate(ranked, 1) if e["priority"] != "HIGH"), None)
    print(f"\nRanked alert list length         : {len(ranked)}")
    print(f"Position of first non-HIGH alert : {first_non_high}  "
          f"({'all alerts are HIGH' if first_non_high is None else 'HIGH alerts occupy positions 1..%d' % (first_non_high - 1)})")
    print("Top 10:")
    for i, e in enumerate(ranked[:10], 1):
        print(f"  {i:>2}. [{e['priority']:<6}] score={e['enhanced_alert_score']:.2f} "
              f"z={e['zscore']:.2f} disk={e['disk_usage_value']}% true={labels.get(key(e))} @ {e['timestamp']}")

    # ------------------------------------------------------------------ G
    title("G. STATISTICAL CONFIDENCE (bootstrap, %d resamples)" % a.bootstrap)
    items = []
    for e in fin:
        if not e["baseline_ready"] or key(e) not in labels:
            continue
        lab = labels[key(e)]
        enh = e["classification"] == "ABNORMAL" and not e["suppressed"]
        sta = e["disk_usage_value"] >= a.static_threshold
        if lab or enh or sta:      # all other readings are TNs; they do not affect P / R / F1
            items.append((lab, enh, sta))
    L = np.array([i[0] for i in items]); E = np.array([i[1] for i in items]); S = np.array([i[2] for i in items])
    rng = np.random.default_rng(a.seed)
    Wt = rng.poisson(1.0, size=(a.bootstrap, len(items))).astype(float)   # Poisson bootstrap

    def prf(P):
        tp = Wt @ (L & P); fp = Wt @ (~L & P); fn = Wt @ (L & ~P)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
        return p, r, f
    pe, re_, fe = prf(E); ps, rs, fs = prf(S)
    ci = lambda x: f"[{np.percentile(x, 2.5):.3f}, {np.percentile(x, 97.5):.3f}]"
    print(f"{'Metric':<12}{'Static 95% CI':>22}{'Enhanced 95% CI':>22}")
    print(f"{'Precision':<12}{ci(ps):>22}{ci(pe):>22}")
    print(f"{'Recall':<12}{ci(rs):>22}{ci(re_):>22}")
    print(f"{'F1':<12}{ci(fs):>22}{ci(fe):>22}")
    d = fe - fs
    print(f"\nPaired F1 difference (Enhanced - Static): mean {d.mean():+.3f}, 95% CI {ci(d)}")
    print(f"Share of resamples where Enhanced F1 > Static F1: {(d > 0).mean() * 100:.1f}%")
    print("Interpretation: if the CI of the F1 difference spans 0, the F1 gain is not statistically")
    print("distinguishable on this sample; the precision/FP-elimination gain is the robust claim.")

    # ------------------------------------------------------------------ H
    title("H. RUNTIME: +/-dt WINDOW vs FULL-CORPUS MATCHING (Problem 4)")
    cls2 = classify([dict(e) for e in base], theta_z=2.0, hard_ceiling=None)
    abn = [e for e in cls2 if e["classification"] == "ABNORMAL"]
    random.seed(a.seed)
    sample = random.sample(abn, min(200, len(abn))) if abn else []
    if sample:
        fail_kw = [k.lower() for k in cfg["correlation"]["failure_keywords"]]
        ben_kw = [k.lower() for k in cfg["correlation"]["benign_keywords"]]
        raws = [c["raw"].lower() for c in ctx]
        t0 = time.perf_counter()
        for e in sample:
            win = get_log_window(ctx, ctx_ts, e["timestamp"], 60)
            classify_priority(win, ben_kw, fail_kw)
        t_win = (time.perf_counter() - t0) / len(sample)
        t0 = time.perf_counter()
        for e in sample[:min(20, len(sample))]:
            # naive: scan every line in the corpus, check time and keywords per line
            hit = None
            for c, rl in zip(ctx, raws):
                if abs((c["timestamp"] - e["timestamp"]).total_seconds()) <= 60:
                    if any(k in rl for k in fail_kw) or any(k in rl for k in ben_kw):
                        hit = True
        t_full = (time.perf_counter() - t0) / min(20, len(sample))
        print(f"Corpus size (log lines)          : {len(ctx)}")
        print(f"Flagged readings sampled         : {len(sample)} of {len(abn)} (at theta_z=2.0)")
        print(f"Windowed (+/-60s, bisect) per alert : {t_win * 1e3:.3f} ms")
        print(f"Full-corpus scan per alert          : {t_full * 1e3:.3f} ms")
        print(f"Speed-up                            : {t_full / t_win:.0f}x")
    else:
        print("No ABNORMAL readings at theta_z=2.0 to benchmark.")


if __name__ == "__main__":
    main()
