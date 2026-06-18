"""
Week 7 — Threshold Sensitivity Analysis
=========================================
Sweeps both governance thresholds across a grid to demonstrate the
governance layer's practical value and identify the stable operating
region of the full pipeline (C4).

Grid:
  conf_threshold  ∈ {0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60}  (7 values)
  contra_threshold ∈ {0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20} (7 values)
  → 49 C4-style configurations evaluated

Also sweeps single-signal sensitivity:
  C1 (confidence only)  : conf_threshold  over 50 values in [0.47, 0.61]
  C3 (NER+contradiction): contra_threshold over 50 values in [0.001, 0.40]

All signals are loaded from the week-5 ablation cache — no re-inference needed.

Output:
  data/week7/threshold_sensitivity.json
  data/week7/threshold_sensitivity_tables.txt
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR    = ROOT / "data/week7"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = ROOT / "data/week5/ablation_cache.json"
LABELS_FILE = ROOT / "data/week3/main_eval/gpt_oss_labels.jsonl"
GEN_DIR     = ROOT / "data/week2/generations/llama2/main"


# ── data loading ──────────────────────────────────────────────────────────────

def load_samples() -> list:
    gt: dict = {}
    for line in LABELS_FILE.read_text().splitlines():
        r = json.loads(line)
        if r["model"] == "llama2" and r["split"] == "main":
            gt[r["id"]] = r["label"]

    samples = []
    for f in sorted(GEN_DIR.glob("q*.json")):
        qid = f.stem
        lbl = gt.get(qid)
        if lbl in ("TRUE", "FALSE"):
            samples.append({"id": qid, "gt_label": lbl})
    return samples


def load_signals(samples: list, cache: dict) -> list:
    """Attach precomputed confidence and contradiction scores to each sample."""
    enriched = []
    for s in samples:
        qid  = s["id"]
        conf = cache.get(f"conf_{qid}", 0.5)
        cont = cache.get(f"contra_ner_{qid}", {"max_score": 0.0})
        enriched.append({
            "id":                 qid,
            "gt_label":           s["gt_label"],
            "conf":               conf,
            "contra_score":       cont["max_score"],
            "is_hall":            s["gt_label"] == "FALSE",
        })
    return enriched


# ── metric computation ────────────────────────────────────────────────────────

def metrics(samples: list, predictions: dict) -> dict:
    tp = fp = tn = fn = 0
    for s in samples:
        pred    = predictions[s["id"]]
        is_hall = s["is_hall"]
        if pred and is_hall:         tp += 1
        elif pred and not is_hall:   fp += 1
        elif not pred and not is_hall: tn += 1
        else:                        fn += 1

    dr   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2 * prec * dr / (prec + dr) if (prec + dr) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "detection_rate":      round(dr,   4),
        "false_positive_rate": round(fpr,  4),
        "precision":           round(prec, 4),
        "f1":                  round(f1,   4),
    }


# ── grid sweep: C4 (both signals) ────────────────────────────────────────────

CONF_GRID  = [0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
CONTRA_GRID = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]


def sweep_c4_grid(samples: list) -> list:
    """
    49-point grid sweep for C4: flag if contra_score ≥ ct OR conf < cf.
    Returns list of {conf_t, contra_t, metrics} sorted by F1.
    """
    results = []
    for cf in CONF_GRID:
        for ct in CONTRA_GRID:
            preds = {s["id"]: (s["contra_score"] >= ct or s["conf"] < cf)
                     for s in samples}
            m = metrics(samples, preds)
            results.append({
                "conf_threshold":   cf,
                "contra_threshold": ct,
                **m,
            })
    return results


# ── single-signal sweeps ──────────────────────────────────────────────────────

def sweep_c1(samples: list, n_steps: int = 60) -> list:
    """Sweep confidence-only threshold from 0.47 to 0.61."""
    thresholds = [round(0.47 + i * (0.61 - 0.47) / (n_steps - 1), 4) for i in range(n_steps)]
    results = []
    for cf in thresholds:
        preds = {s["id"]: s["conf"] < cf for s in samples}
        m = metrics(samples, preds)
        results.append({"conf_threshold": cf, **m})
    return results


def sweep_c3(samples: list, n_steps: int = 60) -> list:
    """Sweep NER+contradiction-only threshold from 0.001 to 0.40."""
    thresholds = [round(0.001 + i * (0.40 - 0.001) / (n_steps - 1), 4) for i in range(n_steps)]
    results = []
    for ct in thresholds:
        preds = {s["id"]: s["contra_score"] >= ct for s in samples}
        m = metrics(samples, preds)
        results.append({"contra_threshold": ct, **m})
    return results


# ── sensitivity metrics ───────────────────────────────────────────────────────

def sensitivity_range(sweep: list, metric: str, threshold_key: str) -> dict:
    """
    Compute how much *metric* changes per unit change in threshold.
    Reports: range of metric values, threshold of maximum gradient,
    stable region (±5% of metric range around best F1).
    """
    best_f1 = max(r["f1"] for r in sweep)
    best_row = next(r for r in sweep if r["f1"] == best_f1)

    vals   = [r[metric] for r in sweep]
    thvals = [r[threshold_key] for r in sweep]

    # gradient between consecutive points
    grads = []
    for i in range(1, len(sweep)):
        dt = thvals[i] - thvals[i - 1]
        dm = vals[i] - vals[i - 1]
        grads.append(abs(dm / dt) if dt != 0 else 0)

    max_grad_idx = grads.index(max(grads))
    max_grad_t   = thvals[max_grad_idx]

    # stable region: configurations within 5% of best F1
    f1_vals     = [r["f1"] for r in sweep]
    f1_thresh   = best_f1 * 0.95
    stable      = [r for r in sweep if r["f1"] >= f1_thresh]
    stable_range = (
        round(min(r[threshold_key] for r in stable), 4),
        round(max(r[threshold_key] for r in stable), 4),
    )

    return {
        "best_f1":            round(best_f1, 4),
        "best_threshold":     best_row[threshold_key],
        "metric_min":         round(min(vals), 4),
        "metric_max":         round(max(vals), 4),
        "metric_range":       round(max(vals) - min(vals), 4),
        "max_gradient_at":    max_grad_t,
        "stable_region":      stable_range,
        "n_stable_configs":   len(stable),
    }


# ── table formatters ──────────────────────────────────────────────────────────

def format_c4_f1_heatmap(grid_results: list) -> str:
    lines = []
    lines.append("  C4 F1 Heatmap (conf_threshold x contra_threshold)")
    lines.append("  " + "─" * 62)
    col_header = "  cf\\ct   " + "".join(f"  {ct:.3f}" for ct in CONTRA_GRID)
    lines.append(col_header)
    lines.append("  " + "─" * 62)
    for cf in CONF_GRID:
        row_vals = [r for r in grid_results if r["conf_threshold"] == cf]
        row_vals.sort(key=lambda r: r["contra_threshold"])
        row_str = f"  {cf:.2f}    " + "".join(f"  {r['f1']:.3f}" for r in row_vals)
        lines.append(row_str)
    lines.append("  " + "─" * 62)
    return "\n".join(lines)


def format_c4_fpr_heatmap(grid_results: list) -> str:
    lines = []
    lines.append("  C4 FPR Heatmap (conf_threshold x contra_threshold)")
    lines.append("  " + "─" * 62)
    col_header = "  cf\\ct   " + "".join(f"  {ct:.3f}" for ct in CONTRA_GRID)
    lines.append(col_header)
    lines.append("  " + "─" * 62)
    for cf in CONF_GRID:
        row_vals = [r for r in grid_results if r["conf_threshold"] == cf]
        row_vals.sort(key=lambda r: r["contra_threshold"])
        row_str = f"  {cf:.2f}    " + "".join(f"  {r['false_positive_rate']:.3f}" for r in row_vals)
        lines.append(row_str)
    lines.append("  " + "─" * 62)
    return "\n".join(lines)


def format_single_sweep(sweep: list, threshold_key: str, title: str) -> str:
    lines = []
    lines.append(f"  {title}")
    lines.append(f"  {'Threshold':<12} {'DR':>7} {'FPR':>7} {'Prec':>7} {'F1':>7}")
    lines.append("  " + "─" * 46)
    step = max(1, len(sweep) // 15)   # print ~15 rows
    for i, r in enumerate(sweep):
        if i % step == 0 or i == len(sweep) - 1:
            lines.append(
                f"  {r[threshold_key]:<12.4f} "
                f"{r['detection_rate']:>6.1%} "
                f"{r['false_positive_rate']:>6.1%} "
                f"{r['precision']:>6.1%} "
                f"{r['f1']:>6.1%}"
            )
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.perf_counter()
    print("=" * 64)
    print("  Week 7 — Threshold Sensitivity Analysis")
    print("=" * 64)

    samples = load_samples()
    cache   = json.loads(CACHE_FILE.read_text())
    samples = load_signals(samples, cache)
    n_hall  = sum(1 for s in samples if s["is_hall"])
    print(f"  {len(samples)} samples  ({n_hall} hallucinatory, {len(samples)-n_hall} truthful)")

    # ── C4 grid sweep ─────────────────────────────────────────────────────────
    print("\n── C4 grid sweep (49 configurations) ──")
    c4_grid = sweep_c4_grid(samples)
    best_c4 = max(c4_grid, key=lambda r: r["f1"])
    best_c4_dr = max(c4_grid, key=lambda r: r["detection_rate"])
    print(f"  Best F1  : f1={best_c4['f1']:.3f}  "
          f"conf_t={best_c4['conf_threshold']}  contra_t={best_c4['contra_threshold']}")
    print(f"  Best DR  : dr={best_c4_dr['detection_rate']:.3f}  "
          f"conf_t={best_c4_dr['conf_threshold']}  contra_t={best_c4_dr['contra_threshold']}")

    # ── C1 single sweep ───────────────────────────────────────────────────────
    print("\n── C1 single-signal sweep (confidence only) ──")
    c1_sweep = sweep_c1(samples)
    best_c1  = max(c1_sweep, key=lambda r: r["f1"])
    print(f"  Best F1  : f1={best_c1['f1']:.3f}  conf_t={best_c1['conf_threshold']}")

    # ── C3 single sweep ───────────────────────────────────────────────────────
    print("\n── C3 single-signal sweep (NER+contradiction only) ──")
    c3_sweep = sweep_c3(samples)
    best_c3  = max(c3_sweep, key=lambda r: r["f1"])
    print(f"  Best F1  : f1={best_c3['f1']:.3f}  contra_t={best_c3['contra_threshold']}")

    # ── sensitivity statistics ────────────────────────────────────────────────
    c1_sens_dr  = sensitivity_range(c1_sweep, "detection_rate", "conf_threshold")
    c1_sens_fpr = sensitivity_range(c1_sweep, "false_positive_rate", "conf_threshold")
    c3_sens_dr  = sensitivity_range(c3_sweep, "detection_rate", "contra_threshold")

    # ── format output ─────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("  Week 7 — Threshold Sensitivity Analysis")
    lines.append("  200 samples · llama2/main · judge: gpt-oss-120b")
    lines.append("=" * 70)
    lines.append("")
    lines.append(format_c4_f1_heatmap(c4_grid))
    lines.append("")
    lines.append(format_c4_fpr_heatmap(c4_grid))
    lines.append("")
    lines.append(f"  Best C4 configuration (by F1)")
    lines.append(f"  conf_threshold={best_c4['conf_threshold']}  "
                 f"contra_threshold={best_c4['contra_threshold']}")
    lines.append(f"  DR={best_c4['detection_rate']:.1%}  FPR={best_c4['false_positive_rate']:.1%}  "
                 f"F1={best_c4['f1']:.1%}")
    lines.append("")
    lines.append(format_single_sweep(c1_sweep, "conf_threshold",
                                     "C1 Confidence-only sweep"))
    lines.append("")
    lines.append(format_single_sweep(c3_sweep, "contra_threshold",
                                     "C3 NER+Contradiction sweep"))
    lines.append("")
    lines.append("  Sensitivity Analysis")
    lines.append("  ──────────────────────────────────────────────────────")
    lines.append(f"  C1 DR range over conf sweep    : "
                 f"{c1_sens_dr['metric_min']:.3f}–{c1_sens_dr['metric_max']:.3f}  "
                 f"(range={c1_sens_dr['metric_range']:.3f})")
    lines.append(f"  C1 FPR range over conf sweep   : "
                 f"{c1_sens_fpr['metric_min']:.3f}–{c1_sens_fpr['metric_max']:.3f}")
    lines.append(f"  C1 stable region (≥95% of best F1): "
                 f"conf ∈ [{c1_sens_dr['stable_region'][0]}, "
                 f"{c1_sens_dr['stable_region'][1]}]  "
                 f"({c1_sens_dr['n_stable_configs']} configs)")
    lines.append(f"  C3 DR range over contra sweep   : "
                 f"{c3_sens_dr['metric_min']:.3f}–{c3_sens_dr['metric_max']:.3f}")
    lines.append(f"  C3 stable region (≥95% of best F1): "
                 f"contra ∈ [{c3_sens_dr['stable_region'][0]}, "
                 f"{c3_sens_dr['stable_region'][1]}]  "
                 f"({c3_sens_dr['n_stable_configs']} configs)")
    lines.append("")
    lines.append("  Governance Layer Implications")
    lines.append("  ─────────────────────────────")

    # find configurations where FPR < 0.50 (production-viable)
    prod_viable = [r for r in c4_grid if r["false_positive_rate"] < 0.50]
    if prod_viable:
        best_viable = max(prod_viable, key=lambda r: r["f1"])
        lines.append(f"  Production-viable (FPR < 50%): {len(prod_viable)} C4 configurations")
        lines.append(f"  Best viable: conf_t={best_viable['conf_threshold']}  "
                     f"contra_t={best_viable['contra_threshold']}  "
                     f"→ DR={best_viable['detection_rate']:.1%}  FPR={best_viable['false_positive_rate']:.1%}  "
                     f"F1={best_viable['f1']:.1%}")
    else:
        best_viable = None
        lines.append("  No C4 configuration achieves FPR < 50% on this eval set.")
        lines.append("  Governance-layer override is critical for production deployment.")

    lines.append("=" * 70)
    table_str = "\n".join(lines)
    print("\n" + table_str)

    # ── save ──────────────────────────────────────────────────────────────────
    elapsed = round(time.perf_counter() - t_start, 3)
    output = {
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "n_samples":   len(samples),
        "grids": {
            "conf_values":   CONF_GRID,
            "contra_values": CONTRA_GRID,
        },
        "c4_grid":      c4_grid,
        "best_c4_f1":   best_c4,
        "best_c4_dr":   best_c4_dr,
        "best_viable":  best_viable,
        "c1_sweep":     c1_sweep,
        "c3_sweep":     c3_sweep,
        "best_c1":      best_c1,
        "best_c3":      best_c3,
        "sensitivity": {
            "c1_dr":  c1_sens_dr,
            "c1_fpr": c1_sens_fpr,
            "c3_dr":  c3_sens_dr,
        },
    }

    (OUT_DIR / "threshold_sensitivity.json").write_text(json.dumps(output, indent=2))
    (OUT_DIR / "threshold_sensitivity_tables.txt").write_text(table_str)

    print(f"\n  Saved → {OUT_DIR}/threshold_sensitivity.json")
    print(f"  Saved → {OUT_DIR}/threshold_sensitivity_tables.txt")
    print(f"  Total elapsed: {elapsed}s")


if __name__ == "__main__":
    main()
