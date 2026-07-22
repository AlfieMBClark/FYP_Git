"""
evaluate.py
-----------
Trajectory-prediction accuracy of every trained model on the held-out 2023 data.

The ADE/FDE figures quoted so far (in the training logs and in the dashboard's
model panel) are *validation* numbers — data that was used to select the
checkpoint. This script reports the same metrics on the 2023 database, which no
model has ever been trained or early-stopped on, and adds the two breakdowns the
project scope promised but never produced:

  * error vs prediction horizon (step 1..10) — where a recurrent model's drift
    compounds and an attention model's does not, this is the plot that shows it;
  * ADE per ship-type group, and on seen vs unseen vessels.

Decoding is fully autoregressive: the decoder is fed its own previous output,
never the ground truth. Teacher-forced error would be optimistic and is not what
the deployed system does.

Metrics
    ADE  average displacement error — mean great-circle error over the 10 steps
    FDE  final displacement error   — error at step 10
    both in km, which is the unit an operator can actually reason about.

Usage
-----
python evaluate.py                       # every model with a checkpoint on disk
python evaluate.py --models transformer gru
"""

import argparse
import csv
import os

import numpy as np

from common import (
    RESULTS_DIR, SEQ_ENC, SEQ_DEC, GROUP_NAMES,
    available_models, load_model, ar_predict, step_errors_km,
    use_report_style, colour_for,
)

_HERE      = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(_HERE, "eval_set.npz")


def summarise(err_km: np.ndarray) -> dict:
    """err_km : (N, SEQ_DEC) -> the headline numbers."""
    per_window_ade = err_km.mean(axis=1)
    return {
        "n":          int(len(err_km)),
        "ade_km":     float(per_window_ade.mean()),
        "ade_median": float(np.median(per_window_ade)),
        "ade_p90":    float(np.percentile(per_window_ade, 90)),
        "fde_km":     float(err_km[:, -1].mean()),
        "step1_km":   float(err_km[:, 0].mean()),
        "rmse_km":    float(np.sqrt((err_km ** 2).mean())),
    }


def main():
    ap = argparse.ArgumentParser(description="Held-out trajectory accuracy, all models.")
    ap.add_argument("--eval-set", default=DEFAULT_IN)
    ap.add_argument("--models",   nargs="+", default=None,
                    help="Registry keys (default: all with a checkpoint).")
    ap.add_argument("--limit",    type=int, default=0,
                    help="Evaluate only the first N windows (0 = all).")
    ap.add_argument("--batch",    type=int, default=256)
    ap.add_argument("--device",   default=None)
    args = ap.parse_args()

    if not os.path.exists(args.eval_set):
        raise SystemExit(f"Missing {args.eval_set} — run build_eval_set.py first.")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    d     = np.load(args.eval_set, allow_pickle=True)
    W     = d["windows"]
    group = d["group"]
    seen  = d["seen"]
    if args.limit:
        W, group, seen = W[: args.limit], group[: args.limit], seen[: args.limit]

    enc    = np.ascontiguousarray(W[:, :SEQ_ENC, :])
    actual = W[:, SEQ_ENC:, :]

    print(f"\n  Evaluation set : {len(W):,} windows "
          f"({int(seen.sum()):,} seen vessels / {int((~seen).sum()):,} unseen)")
    print(f"  Device         : {device}")

    models   = available_models(args.models)
    rows     = []
    horizons = {}

    for key, meta in models:
        print(f"\n  ── {meta['label']} ({key}) ──")
        model = load_model(key, device)
        mu, _ = ar_predict(model, enc, device, batch_size=args.batch, progress=True)
        err   = step_errors_km(actual, mu)               # (N, SEQ_DEC) km
        horizons[key] = err.mean(axis=0)

        overall = summarise(err)
        print(f"    ADE {overall['ade_km']:.3f} km   FDE {overall['fde_km']:.3f} km   "
              f"(median ADE {overall['ade_median']:.3f}, p90 {overall['ade_p90']:.3f})")
        rows.append({"model": key, "label": meta["label"], "subset": "all", **overall})

        for name, mask in (("seen_vessels", seen), ("unseen_vessels", ~seen)):
            if mask.sum() == 0:
                continue
            s = summarise(err[mask])
            print(f"    {name:<15} ADE {s['ade_km']:.3f} km   FDE {s['fde_km']:.3f} km "
                  f"(n={s['n']:,})")
            rows.append({"model": key, "label": meta["label"], "subset": name, **s})

        for g in range(len(GROUP_NAMES)):
            mask = group == g
            if mask.sum() < 30:      # too few windows to quote a number
                continue
            s = summarise(err[mask])
            rows.append({"model": key, "label": meta["label"],
                         "subset": f"type_{GROUP_NAMES[g]}", **s})

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── outputs ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "prediction_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    plt = use_report_style()

    # error vs horizon
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    steps = np.arange(1, SEQ_DEC + 1)
    for key, meta in models:
        ax.plot(steps, horizons[key], marker="o", ms=3.5, lw=1.8,
                color=colour_for(key), label=meta["label"])
    ax.set_xlabel("Prediction horizon (AIS pings ahead)")
    ax.set_ylabel("Mean displacement error (km)")
    ax.set_title("Error growth over the prediction horizon — 2023 hold-out")
    ax.set_xticks(steps)
    ax.legend()
    fig.savefig(os.path.join(RESULTS_DIR, "error_vs_horizon.png"))
    plt.close(fig)

    # ADE by ship type
    groups_present = [g for g in range(len(GROUP_NAMES))
                      if (group == g).sum() >= 30]
    if groups_present:
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        width = 0.8 / len(models)
        x = np.arange(len(groups_present))
        for i, (key, meta) in enumerate(models):
            vals = [next(r["ade_km"] for r in rows
                         if r["model"] == key and r["subset"] == f"type_{GROUP_NAMES[g]}")
                    for g in groups_present]
            ax.bar(x + i * width, vals, width, label=meta["label"], color=colour_for(key))
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels([f"{GROUP_NAMES[g]}\n(n={int((group == g).sum()):,})"
                            for g in groups_present], fontsize=8)
        ax.set_ylabel("ADE (km)")
        ax.set_title("Prediction error by vessel type — 2023 hold-out")
        ax.legend()
        fig.savefig(os.path.join(RESULTS_DIR, "ade_by_shiptype.png"))
        plt.close(fig)

    # seen vs unseen vessels
    if seen.any() and (~seen).any():
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        x = np.arange(2)
        width = 0.8 / len(models)
        for i, (key, meta) in enumerate(models):
            vals = [next(r["ade_km"] for r in rows
                         if r["model"] == key and r["subset"] == s)
                    for s in ("seen_vessels", "unseen_vessels")]
            ax.bar(x + i * width, vals, width, label=meta["label"], color=colour_for(key))
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels([f"Vessel seen in training\n(n={int(seen.sum()):,})",
                            f"Vessel never seen\n(n={int((~seen).sum()):,})"], fontsize=9)
        ax.set_ylabel("ADE (km)")
        ax.set_title("Generalisation to unseen vessels")
        ax.legend()
        fig.savefig(os.path.join(RESULTS_DIR, "ade_seen_vs_unseen.png"))
        plt.close(fig)

    # ── headline table ────────────────────────────────────────────────────────
    print(f"\n  ══ Held-out results (2023) ══")
    print(f"  {'Model':<22}{'ADE km':>9}{'FDE km':>9}{'Unseen ADE':>13}")
    for key, meta in models:
        a = next(r for r in rows if r["model"] == key and r["subset"] == "all")
        u = next((r for r in rows if r["model"] == key
                  and r["subset"] == "unseen_vessels"), None)
        u_txt = f"{u['ade_km']:.3f}" if u else "—"
        print(f"  {meta['label']:<22}{a['ade_km']:>9.3f}{a['fde_km']:>9.3f}{u_txt:>13}")

    print(f"\n  CSV     -> {csv_path}")
    print(f"  Figures -> {RESULTS_DIR}/error_vs_horizon.png, ade_by_shiptype.png, "
          f"ade_seen_vs_unseen.png\n")


if __name__ == "__main__":
    main()
