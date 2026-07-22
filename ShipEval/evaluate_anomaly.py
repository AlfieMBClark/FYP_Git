"""
evaluate_anomaly.py
-------------------
Quantitative evaluation of the prediction-driven anomaly detector: precision,
recall, F1, ROC-AUC, PR-AUC, and the false-alarm rate on clean traffic.

This is the script that turns Objective 3 from *demonstrated* into *evidenced*.
Until now the detector could be shown working on a map; it could not be scored,
because there was no labelled data. inject_anomalies.py supplies the labels.

How a window is scored
----------------------
The model rolls out 10 steps autoregressively and, being probabilistic, emits a
mean and a standard deviation per step. The anomaly score is how many sigma the
*observed* track sat from that prediction. Several scoring rules are compared,
and the comparison is itself a finding worth reporting:

    mean_z         mean |z| over all 10 steps and all 5 predicted features
    max_z          the single worst step (a brief, sharp manoeuvre shows up
                   here but gets diluted in mean_z)
    pos_z_mean     mean |z| over LAT/LON only — position, ignoring self-
                   reported speed and course, which the interviewees said
                   they trust least
    pos_z_peak     peak per-step LAT/LON |z| — the raw quantity the dashboard
                   and sim engine compute
    pos_z_peak_cal pos_z_peak after the empirical calibration the deployed
                   system applies (see below). This is the shipped score, so
                   it is the one the 3-sigma threshold is judged against.
    ade_km         plain mean displacement error in km, ignoring sigma
                   entirely. This is the ablation: if ade_km matches the
                   z-scores, the learned variance head earns nothing and could
                   be dropped. If it does not, the probabilistic head is
                   justified.

Calibration
-----------
The raw z-score is not calibrated: the model's predicted sigma is far wider
than its actual error, so raw z is compressed towards zero and a literal
3-sigma threshold fires on nothing. sim_engine.py works around this by dividing
the raw score by (p90 / 1.2816) — rescaling so the 90th percentile of observed
scores lands where a standard normal's would. This script fits that same scale
factor but fits it *on the clean windows only*, which is the defensible version:
the calibration is learned from normal traffic, exactly like the model itself.
The fitted factor is reported — it quantifies how over-dispersed sigma is, and
it is the reason a raw 3-sigma threshold was never going to work.

Note that rescaling is monotone, so ROC-AUC and PR-AUC are identical before and
after calibration. Calibration does not make the detector better; it makes the
*threshold* mean something.

Because injection never touches the encoder, the model's prediction is identical
for a source window's clean and anomalous variants — so inference runs once per
source and both variants are scored against it. Clean and injected are a matched
pair; the only thing that differs is what the vessel actually did next.

Usage
-----
python evaluate_anomaly.py
python evaluate_anomaly.py --models transformer gru tcn --threshold 3.0
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from common import (
    RESULTS_DIR, SEQ_ENC, N_DEC,
    available_models, load_model, ar_predict, step_errors_km,
    use_report_style, colour_for,
)

_HERE      = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(_HERE, "injected_set.npz")

SCORERS = ["mean_z", "max_z", "pos_z_mean", "pos_z_peak", "pos_z_peak_cal", "ade_km"]
PRIMARY = "pos_z_peak_cal"   # the rule the deployed system actually ships
NORMAL_P90 = 1.2816          # 90th percentile of a standard normal


# ── metric primitives (no sklearn in this environment) ────────────────────────

def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based (Mann-Whitney U) AUC — ties handled by average ranks."""
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within tied score groups
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def pr_curve(y: np.ndarray, s: np.ndarray):
    """Precision/recall at every distinct threshold, plus average precision."""
    order = np.argsort(-s, kind="mergesort")
    y_s   = y[order]
    tp    = np.cumsum(y_s)
    fp    = np.cumsum(1 - y_s)
    prec  = tp / np.maximum(tp + fp, 1)
    rec   = tp / max(int(y.sum()), 1)
    ap    = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    return prec, rec, s[order], ap


def roc_curve(y: np.ndarray, s: np.ndarray):
    order = np.argsort(-s, kind="mergesort")
    y_s   = y[order]
    tp    = np.cumsum(y_s)
    fp    = np.cumsum(1 - y_s)
    tpr   = tp / max(int(y.sum()), 1)
    fpr   = fp / max(int((1 - y).sum()), 1)
    return fpr, tpr


def at_threshold(y: np.ndarray, s: np.ndarray, t: float) -> dict:
    pred = (s >= t).astype(np.int8)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr  = fp / (fp + tn) if fp + tn else 0.0
    return {"threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "false_alarm_rate": fpr}


def best_f1(y: np.ndarray, s: np.ndarray) -> dict:
    """Sweep the threshold and keep the F1-optimal operating point."""
    cands = np.unique(np.quantile(s, np.linspace(0, 1, 400)))
    best = {"f1": -1.0}
    for t in cands:
        m = at_threshold(y, s, t)
        if m["f1"] > best["f1"]:
            best = m
    return best


# ── scoring ───────────────────────────────────────────────────────────────────

def score_windows(actual, mu, sigma, clean_mask):
    """Every scoring rule for a batch. actual/mu/sigma are (N, SEQ_DEC, N_DEC)
    in normalised space; the km score is denormalised internally.

    clean_mask selects the label-0 windows, which are the only ones the sigma
    calibration is allowed to see — fitting the scale on anomalies too would
    let the positives shift their own threshold.
    """
    z = np.abs(actual[:, :, :N_DEC] - mu) / (sigma + 1e-8)   # (N, T, 5)

    pos_step = z[:, :, :2].mean(axis=2)      # (N, T) — the sim engine's raw_z
    pos_peak = pos_step.max(axis=1)

    # z_scale: what the deployed system divides by. Fitted on clean traffic.
    p90     = float(np.percentile(pos_peak[clean_mask], 90))
    z_scale = p90 / NORMAL_P90 if p90 > 1e-9 else 1.0

    scores = {
        "mean_z":         z.mean(axis=(1, 2)),
        "max_z":          z.mean(axis=2).max(axis=1),
        "pos_z_mean":     pos_step.mean(axis=1),
        "pos_z_peak":     pos_peak,
        "pos_z_peak_cal": pos_peak / z_scale,
        "ade_km":         step_errors_km(actual, mu).mean(axis=1),
    }
    return scores, z_scale, pos_peak[clean_mask]


def main():
    ap = argparse.ArgumentParser(description="Anomaly-detection metrics.")
    ap.add_argument("--injected", default=DEFAULT_IN)
    ap.add_argument("--models",   nargs="+", default=None)
    ap.add_argument("--threshold", type=float, default=3.0,
                    help="The sigma threshold the system currently ships with "
                         "(predict.py and the dashboard both default to 3.0).")
    ap.add_argument("--batch",  type=int, default=256)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.injected):
        raise SystemExit(f"Missing {args.injected} — run inject_anomalies.py first.")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    d      = np.load(args.injected, allow_pickle=True)
    W      = d["windows"]
    y      = d["label"].astype(np.int8)
    atype  = d["atype"]
    sev    = d["severity"]
    src    = d["source_id"]

    actual = W[:, SEQ_ENC:, :]

    # One inference per source window: every variant of a source shares its
    # encoder, so they share the prediction too. Assert it rather than trust it.
    uniq, first = np.unique(src, return_index=True)
    enc_by_src  = np.ascontiguousarray(W[first, :SEQ_ENC, :])
    probe = min(200, len(W))
    for i in np.random.default_rng(0).choice(len(W), probe, replace=False):
        s_pos = int(np.searchsorted(uniq, src[i]))
        assert np.array_equal(W[i, :SEQ_ENC, :], enc_by_src[s_pos]), \
            "encoder differs within a source group — injection touched the context"
    src_pos = np.searchsorted(uniq, src)     # window -> row in enc_by_src

    print(f"\n  Injected set : {len(W):,} windows  "
          f"({int((y == 0).sum()):,} clean / {int((y == 1).sum()):,} anomalous)")
    print(f"  Sources      : {len(uniq):,} (one inference each)")
    print(f"  Device       : {device}")

    models = available_models(args.models)
    rows, curves, per_model_scores, calib = [], {}, {}, {}
    clean = y == 0

    for key, meta in models:
        print(f"\n  ── {meta['label']} ({key}) ──")
        model = load_model(key, device)
        mu_s, sig_s = ar_predict(model, enc_by_src, device,
                                 batch_size=args.batch, progress=True)
        mu    = mu_s[src_pos]           # broadcast the source prediction to its variants
        sigma = sig_s[src_pos]
        scores, z_scale, clean_raw = score_windows(actual, mu, sigma, clean)
        per_model_scores[key] = scores
        calib[key] = z_scale

        # How badly is sigma calibrated? A well-calibrated model would need no
        # rescaling at all (z_scale = 1). Report it, because it is the reason
        # the raw 3-sigma default could never have fired.
        print(f"     sigma calibration: raw peak-z on clean traffic has "
              f"p50={np.median(clean_raw):.2f}, p90={np.percentile(clean_raw, 90):.2f} "
              f"→ z_scale {z_scale:.3f}")
        if z_scale < 0.5:
            print(f"     (sigma is over-dispersed ≈{1 / z_scale:.1f}x — the model's "
                  f"stated uncertainty is far wider than its actual error)")

        for scorer in SCORERS:
            s   = scores[scorer]
            auc = roc_auc(y, s)
            _, _, _, ap_ = pr_curve(y, s)
            bf  = best_f1(y, s)
            row = {"model": key, "label": meta["label"], "scorer": scorer,
                   "z_scale": z_scale if scorer == PRIMARY else "",
                   "roc_auc": auc, "pr_auc": ap_,
                   "best_f1": bf["f1"], "best_f1_threshold": bf["threshold"],
                   "precision_at_best": bf["precision"], "recall_at_best": bf["recall"],
                   "false_alarm_at_best": bf["false_alarm_rate"]}

            # The shipped operating point is only meaningful on the calibrated
            # sigma score — a "3 sigma" cut on a km error or on an uncalibrated
            # z means nothing.
            if scorer == PRIMARY:
                op = at_threshold(y, s, args.threshold)
                row.update({
                    f"precision_at_{args.threshold:g}s": op["precision"],
                    f"recall_at_{args.threshold:g}s":    op["recall"],
                    f"f1_at_{args.threshold:g}s":        op["f1"],
                    f"false_alarm_at_{args.threshold:g}s": op["false_alarm_rate"],
                })
            rows.append(row)

            marker = " *" if scorer == PRIMARY else "  "
            print(f"   {marker}{scorer:<15} ROC-AUC {auc:.3f}   PR-AUC {ap_:.3f}   "
                  f"best F1 {bf['f1']:.3f} @ {bf['threshold']:.2f}")

        s_primary = scores[PRIMARY]
        curves[key] = {
            "roc": roc_curve(y, s_primary),
            "pr":  pr_curve(y, s_primary)[:2],
            "op":  at_threshold(y, s_primary, args.threshold),
            "best": best_f1(y, s_primary),
            "auc": roc_auc(y, s_primary),
        }

        op = curves[key]["op"]
        print(f"     at {args.threshold:g}σ (the shipped default): "
              f"precision {op['precision']:.3f}  recall {op['recall']:.3f}  "
              f"F1 {op['f1']:.3f}  false alarms {100 * op['false_alarm_rate']:.1f}% "
              f"of clean traffic")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── best model on the primary scorer drives the breakdown plots ───────────
    best_key  = max(curves, key=lambda k: curves[k]["auc"])
    best_meta = dict(models)[best_key]

    # per-type recall at the operating threshold, for every model
    types = [t for t in dict.fromkeys(atype) if t != "clean"]
    type_recall = defaultdict(dict)
    for key, _ in models:
        s = per_model_scores[key][PRIMARY]
        t_op = curves[key]["op"]["threshold"]
        for t in types:
            m = atype == t
            type_recall[key][t] = float((s[m] >= t_op).mean()) if m.any() else np.nan

    # recall vs severity for the best model.
    # Levels are ordered weakest -> strongest, which for speed_drop means
    # *descending* (a 0.5x slowdown is mild; stopping dead is not).
    STRONGER_IS_LOWER = {"speed_drop"}
    sev_recall = defaultdict(list)
    s_best = per_model_scores[best_key][PRIMARY]
    t_best = curves[best_key]["op"]["threshold"]
    for t in types:
        levels = sorted(np.unique(sev[atype == t]),
                        reverse=t in STRONGER_IS_LOWER)
        for lvl in levels:
            m = (atype == t) & (sev == lvl)
            sev_recall[t].append((float(lvl), float((s_best[m] >= t_best).mean()),
                                  int(m.sum())))

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "anomaly_metrics.csv")
    fields = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    type_csv = os.path.join(RESULTS_DIR, "anomaly_recall_by_type.csv")
    with open(type_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "anomaly_type", "severity", "n", "recall"])
        for key, _ in models:
            for t in types:
                w.writerow([key, t, "all", int((atype == t).sum()),
                            f"{type_recall[key][t]:.4f}"])
        for t, entries in sev_recall.items():
            for lvl, rec, n in entries:
                w.writerow([best_key, t, lvl, n, f"{rec:.4f}"])

    # ── plots ─────────────────────────────────────────────────────────────────
    plt = use_report_style()

    # ROC + PR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for key, meta in models:
        fpr, tpr = curves[key]["roc"]
        ax1.plot(fpr, tpr, lw=1.8, color=colour_for(key),
                 label=f"{meta['label']}  (AUC {curves[key]['auc']:.3f})")
        prec, rec = curves[key]["pr"]
        ax2.plot(rec, prec, lw=1.8, color=colour_for(key), label=meta["label"])
    ax1.plot([0, 1], [0, 1], ls="--", lw=1, color="#999", label="chance")
    ax1.set_xlabel("False-alarm rate (clean traffic flagged)")
    ax1.set_ylabel("Detection rate (anomalies caught)")
    ax1.set_title(f"ROC — anomaly detection ({PRIMARY})")
    ax1.legend(fontsize=8)
    ax2.axhline((y == 1).mean(), ls="--", lw=1, color="#999", label="chance")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision–recall")
    ax2.legend(fontsize=8)
    fig.savefig(os.path.join(RESULTS_DIR, "anomaly_roc_pr.png"))
    plt.close(fig)

    # score distributions, best model
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    s = per_model_scores[best_key][PRIMARY]
    hi = float(np.percentile(s, 99.5))
    bins = np.linspace(0, hi, 60)
    ax.hist(s[y == 0], bins=bins, alpha=0.65, label="clean", color="#4C72B0")
    ax.hist(s[y == 1], bins=bins, alpha=0.65, label="anomalous", color="#C44E52")
    ax.axvline(args.threshold, color="k", ls="--", lw=1.2,
               label=f"shipped threshold ({args.threshold:g}σ)")
    ax.axvline(t_best, color="#55A868", ls=":", lw=1.6,
               label=f"F1-optimal ({t_best:.2f}σ)")
    ax.set_xlabel(f"Anomaly score ({PRIMARY})")
    ax.set_ylabel("Windows")
    ax.set_title(f"Score separation — {best_meta['label']}")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(RESULTS_DIR, "anomaly_score_distribution.png"))
    plt.close(fig)

    # recall by anomaly type
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    x = np.arange(len(types))
    width = 0.8 / len(models)
    for i, (key, meta) in enumerate(models):
        ax.bar(x + i * width, [type_recall[key][t] for t in types], width,
               label=meta["label"], color=colour_for(key))
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall at the operating threshold")
    ax.set_title("What kind of anomaly gets caught")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(RESULTS_DIR, "anomaly_recall_by_type.png"))
    plt.close(fig)

    # recall vs severity, best model
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for t, entries in sev_recall.items():
        xs = [e[0] for e in entries]
        ys = [e[1] for e in entries]
        # normalise the x-axis to "rank of severity" so six different units
        # (degrees, speed factors, km, steps) can share one plot
        ax.plot(range(len(xs)), ys, marker="o", ms=4, lw=1.6,
                label=f"{t}  ({xs[0]:g}→{xs[-1]:g})")
    ax.set_xlabel("Severity level (weakest → strongest, units differ per type)")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Sensitivity floor — {best_meta['label']} at "
                 f"{t_best:.2f}σ")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(RESULTS_DIR, "anomaly_recall_vs_severity.png"))
    plt.close(fig)

    # ── console summary ───────────────────────────────────────────────────────
    print(f"\n  ══ Detection at the shipped {args.threshold:g}σ threshold "
          f"(scorer: {PRIMARY}) ══")
    print(f"  {'Model':<22}{'AUC':>7}{'Prec':>7}{'Recall':>8}{'F1':>7}"
          f"{'False alarms':>14}{'z_scale':>9}")
    for key, meta in models:
        op = curves[key]["op"]
        print(f"  {meta['label']:<22}{curves[key]['auc']:>7.3f}{op['precision']:>7.3f}"
              f"{op['recall']:>8.3f}{op['f1']:>7.3f}"
              f"{100 * op['false_alarm_rate']:>13.1f}%{calib[key]:>9.3f}")

    print(f"\n  Best model on ROC-AUC: {best_meta['label']}")
    bf = curves[best_key]["best"]
    print(f"  F1-optimal threshold : {bf['threshold']:.2f}σ  "
          f"(F1 {bf['f1']:.3f}, precision {bf['precision']:.3f}, "
          f"recall {bf['recall']:.3f}, false alarms "
          f"{100 * bf['false_alarm_rate']:.1f}%)")
    print(f"  Confusion at {args.threshold:g}σ: "
          f"TP {curves[best_key]['op']['tp']:,}  FP {curves[best_key]['op']['fp']:,}  "
          f"FN {curves[best_key]['op']['fn']:,}  TN {curves[best_key]['op']['tn']:,}")

    print(f"\n  Recall by anomaly type ({best_meta['label']}):")
    for t in types:
        print(f"    {t:<18} {type_recall[best_key][t]:.3f}")

    print(f"\n  CSV     -> {csv_path}")
    print(f"             {type_csv}")
    print(f"  Figures -> {RESULTS_DIR}/anomaly_roc_pr.png, "
          f"anomaly_score_distribution.png,\n"
          f"             anomaly_recall_by_type.png, anomaly_recall_vs_severity.png\n")


if __name__ == "__main__":
    main()
