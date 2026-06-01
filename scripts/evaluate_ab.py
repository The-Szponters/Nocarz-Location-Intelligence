"""Evaluate the A/B experiment from the microservice log.

Reads logs/predictions.jsonl, joins ground truth on listing_id, and compares
model A vs B: per-model RMSE/MAE/R2, statistical significance (Mann-Whitney U
+ Welch t-test on absolute errors; paired Wilcoxon if forced /a-/b traffic is
present), a bootstrap CI on the RMSE gap, plots, and a verdict.

Functions are importable by notebooks/03_ab_evaluation.ipynb.
Usage:  python scripts/evaluate_ab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nocarz.features import LOGS_DIR, PROCESSED_DIR, REPORTS_DIR  # noqa: E402

LOG_PATH = LOGS_DIR / "predictions.jsonl"
GT_PATH = PROCESSED_DIR / "ground_truth.csv"
FIG_DIR = REPORTS_DIR / "figures"
REPORT_PATH = REPORTS_DIR / "ab_report.md"
MODEL_LABELS = {"a": "A (baseline / district-mean)", "b": "B (HGB)"}


# --- data loading ----------------------------------------------------------
def load_log(path: Path = LOG_PATH) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    return df


def join_truth(log_df: pd.DataFrame, gt_path: Path = GT_PATH) -> pd.DataFrame:
    gt = pd.read_csv(gt_path)
    df = log_df.merge(gt, on="listing_id", how="inner")
    df["abs_err"] = (df["predicted_annual_revenue"] - df["true_annual_revenue"]).abs()
    df["sq_err"] = (df["predicted_annual_revenue"] - df["true_annual_revenue"]) ** 2
    return df


def natural_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """Rows from the live A/B split (hash-routed main endpoint only)."""
    return df[(df["assignment_reason"] == "hash") &
              (df["endpoint"] == "/predict_revenue")].copy()


# --- metrics ---------------------------------------------------------------
def per_model_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for role, g in df.groupby("assigned_model"):
        yt, yp = g["true_annual_revenue"], g["predicted_annual_revenue"]
        rows.append({
            "model": role,
            "n": len(g),
            "rmse": mean_squared_error(yt, yp) ** 0.5,
            "mae": mean_absolute_error(yt, yp),
            "r2": r2_score(yt, yp) if len(g) > 1 else float("nan"),
            "median_ae": g["abs_err"].median(),
        })
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def significance(df: pd.DataFrame) -> dict:
    """Independent tests on absolute errors between the two live groups."""
    err_a = df[df["assigned_model"] == "a"]["abs_err"].to_numpy()
    err_b = df[df["assigned_model"] == "b"]["abs_err"].to_numpy()
    out = {"n_a": len(err_a), "n_b": len(err_b)}
    if len(err_a) and len(err_b):
        out["mwu_p"] = float(stats.mannwhitneyu(err_a, err_b, alternative="two-sided")[1])
        out["welch_p"] = float(stats.ttest_ind(err_a, err_b, equal_var=False)[1])
        out["mean_ae_a"] = float(err_a.mean())
        out["mean_ae_b"] = float(err_b.mean())
    return out


def bootstrap_rmse_gap(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple:
    """95% CI for RMSE(A) - RMSE(B), resampling within each group."""
    rng = np.random.default_rng(seed)
    a = df[df["assigned_model"] == "a"]["sq_err"].to_numpy()
    b = df[df["assigned_model"] == "b"]["sq_err"].to_numpy()
    if not len(a) or not len(b):
        return (float("nan"), float("nan"))
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        gaps[i] = np.sqrt(ra.mean()) - np.sqrt(rb.mean())
    return (float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5)))


def paired_test(df_all: pd.DataFrame) -> dict | None:
    """Wilcoxon on per-listing abs-error difference, using forced /a-/b logs."""
    fa = df_all[df_all["endpoint"] == "/predict_revenue/a"][["listing_id", "abs_err"]]
    fb = df_all[df_all["endpoint"] == "/predict_revenue/b"][["listing_id", "abs_err"]]
    if fa.empty or fb.empty:
        return None
    merged = fa.merge(fb, on="listing_id", suffixes=("_a", "_b"))
    if len(merged) < 5:
        return None
    diff = merged["abs_err_a"] - merged["abs_err_b"]
    return {
        "n_pairs": len(merged),
        "mean_ae_a": float(merged["abs_err_a"].mean()),
        "mean_ae_b": float(merged["abs_err_b"].mean()),
        "wilcoxon_p": float(stats.wilcoxon(diff)[1]),
        "paired_t_p": float(stats.ttest_rel(merged["abs_err_a"], merged["abs_err_b"])[1]),
    }


# --- verdict ---------------------------------------------------------------
def decide(metrics: pd.DataFrame, ci_gap: tuple, sig: dict, paired: dict | None) -> str:
    """Weigh the full evidence. The paired test (same listings via forced /a-/b)
    is the most powerful and is preferred when available; the independent
    Mann-Whitney corroborates it; the bootstrap RMSE CI is reported as a caveat
    because RMSE is dominated by the heavy error tail at this sample size."""
    m = metrics.set_index("model")
    if "a" not in m.index or "b" not in m.index:
        return "Niewystarczające dane (brak ruchu dla obu modeli)."
    rmse_a, rmse_b = m.loc["a", "rmse"], m.loc["b", "rmse"]
    mae_a, mae_b = m.loc["a", "mae"], m.loc["b", "mae"]
    b_better = (rmse_b < rmse_a) and (mae_b < mae_a)

    paired_sig = bool(paired and paired["wilcoxon_p"] < 0.05
                      and paired["mean_ae_b"] < paired["mean_ae_a"])
    indep_sig = sig.get("mwu_p", 1.0) < 0.05
    rmse_ci_sig = ci_gap[0] > 0
    caveat = ("" if rmse_ci_sig else
              f" (uwaga: luka RMSE nieistotna, 95% CI = [{ci_gap[0]:,.0f}, "
              f"{ci_gap[1]:,.0f}] — RMSE zdominowane przez ciężki ogon błędów; "
              f"przewaga B dotyczy ofert typowych)")

    if b_better and (paired_sig or indep_sig):
        tests = []
        if paired_sig:
            tests.append(f"parowany Wilcoxon p={paired['wilcoxon_p']:.2g}")
        if indep_sig:
            tests.append(f"Mann-Whitney p={sig['mwu_p']:.2g}")
        return (f"WYGRYWA model B (HGB): MAE {mae_b:,.0f} < {mae_a:,.0f} EUR, "
                f"RMSE {rmse_b:,.0f} < {rmse_a:,.0f}; różnica istotna ({', '.join(tests)}). "
                f"Rekomendacja: wdrożyć B{caveat}.")
    if b_better:
        return (f"Model B ma niższe MAE/RMSE, ale różnica nieistotna statystycznie. "
                f"Rekomendacja: zebrać więcej danych lub pozostać przy modelu A{caveat}.")
    return (f"Model A nie jest gorszy od B (RMSE {rmse_a:,.0f} <= {rmse_b:,.0f}). "
            f"Rekomendacja: prostszy model A.")


# --- plots -----------------------------------------------------------------
def make_plots(df: pd.DataFrame, outdir: Path = FIG_DIR) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1) predicted vs actual per model
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, role in zip(axes, ["a", "b"]):
        g = df[df["assigned_model"] == role]
        ax.scatter(g["true_annual_revenue"], g["predicted_annual_revenue"], s=8, alpha=0.3)
        lim = [0, df["true_annual_revenue"].quantile(0.99)]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set(xlim=lim, ylim=lim, title=MODEL_LABELS[role],
               xlabel="rzeczywisty przychód [EUR]", ylabel="predykcja [EUR]")
    fig.suptitle("Predykcja vs rzeczywistość")
    p = outdir / "ab_pred_vs_actual.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    paths.append(p)

    # 2) absolute error boxplot
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df[df["assigned_model"] == r]["abs_err"] for r in ["a", "b"]]
    ax.boxplot(data, tick_labels=[MODEL_LABELS["a"], MODEL_LABELS["b"]], showfliers=False)
    ax.set(ylabel="błąd bezwzględny [EUR]", title="Rozkład błędu bezwzględnego wg modelu")
    p = outdir / "ab_abs_error_box.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    paths.append(p)

    # 3) RMSE / MAE bars
    m = per_model_metrics(df)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(m)); w = 0.35
    ax.bar(x - w / 2, m["rmse"], w, label="RMSE")
    ax.bar(x + w / 2, m["mae"], w, label="MAE")
    ax.set_xticks(x); ax.set_xticklabels(m["model"].map(MODEL_LABELS))
    ax.set(ylabel="EUR", title="RMSE / MAE wg modelu"); ax.legend()
    p = outdir / "ab_metric_bars.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    paths.append(p)
    return paths


# --- report ----------------------------------------------------------------
def _md_table(metrics: pd.DataFrame) -> str:
    """Render the metrics table as Markdown without the optional tabulate dep."""
    header = "| model | n | RMSE | MAE | R² | mediana AE |"
    sep = "|---|---|---|---|---|---|"
    rows = []
    for _, r in metrics.iterrows():
        rows.append(
            f"| {MODEL_LABELS.get(r['model'], r['model'])} | {int(r['n']):,} | "
            f"{r['rmse']:,.0f} | {r['mae']:,.0f} | {r['r2']:.3f} | {r['median_ae']:,.0f} |"
        )
    return "\n".join([header, sep, *rows])


def write_report(metrics, sig, ci_gap, paired, verdict_text, path: Path = REPORT_PATH):
    lines = ["# Raport z eksperymentu A/B — Nocarz\n",
             f"Źródło: `{LOG_PATH.name}` (ruch produkcyjny, routing hash 50/50).\n",
             "## Metryki per model (ruch naturalny A/B)\n",
             _md_table(metrics), "\n",
             "## Istotność statystyczna (błąd bezwzględny, testy niezależne)\n",
             f"- Mann-Whitney U: p = {sig.get('mwu_p', float('nan')):.4g}",
             f"- Welch t-test: p = {sig.get('welch_p', float('nan')):.4g}",
             f"- Bootstrap 95% CI dla RMSE(A) − RMSE(B): "
             f"[{ci_gap[0]:,.0f}, {ci_gap[1]:,.0f}] EUR\n"]
    if paired:
        lines += ["## Test parowany (wymuszone /a i /b na tych samych ofertach)\n",
                  f"- liczba par: {paired['n_pairs']:,}",
                  f"- średni |błąd| A = {paired['mean_ae_a']:,.0f}, "
                  f"B = {paired['mean_ae_b']:,.0f} EUR",
                  f"- Wilcoxon: p = {paired['wilcoxon_p']:.4g}; "
                  f"t-parowany: p = {paired['paired_t_p']:.4g}\n"]
    lines += ["## Werdykt\n", verdict_text, "\n",
              "## Wykresy\n",
              "![pred vs actual](figures/ab_pred_vs_actual.png)",
              "![abs error](figures/ab_abs_error_box.png)",
              "![metrics](figures/ab_metric_bars.png)"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    log = load_log()
    df_all = join_truth(log)
    nat = natural_traffic(df_all)
    if nat.empty:
        print("No natural A/B traffic in the log. Run simulate_clients.py first.")
        return

    metrics = per_model_metrics(nat)
    sig = significance(nat)
    ci_gap = bootstrap_rmse_gap(nat)
    paired = paired_test(df_all)
    verdict_text = decide(metrics, ci_gap, sig, paired)
    make_plots(nat)
    write_report(metrics, sig, ci_gap, paired, verdict_text)

    print("=== Per-model metrics (natural A/B traffic) ===")
    print(metrics.to_string(index=False))
    print(f"\nMann-Whitney p={sig.get('mwu_p'):.4g}  Welch p={sig.get('welch_p'):.4g}")
    print(f"Bootstrap 95% CI RMSE(A)-RMSE(B): [{ci_gap[0]:,.0f}, {ci_gap[1]:,.0f}]")
    if paired:
        print(f"Paired Wilcoxon p={paired['wilcoxon_p']:.4g} (n={paired['n_pairs']})")
    print(f"\nVERDICT: {verdict_text}")
    print(f"\nReport -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
