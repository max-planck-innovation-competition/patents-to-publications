"""
Paired significance tests on the per-query scores of the patent-to-publication
(cross-corpus) benchmark.

All models are evaluated on the same queries, so we test the per-query score
differences between two models: paired t-test, Wilcoxon signed-rank test and a
bootstrapped 95% confidence interval of the mean difference. Each transformer
model enters at its default pooling from models_config.yaml; BM25 is included.

Outputs (in ./output):
- significance_test.tex: difference of each model to the reference model
  (BGE), paired t-test standard errors in parentheses
- significance_test.csv: all pairwise comparisons with all three tests
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.models_config import (
    get_all_models,
    get_bm25_directory_name,
    get_bm25_display_name,
)


# --- Config ---
SCORES_DIR = Path("./output/ranking")
DEST_DIR = Path("./output")
DEST_DIR.mkdir(parents=True, exist_ok=True)

# Model the others are compared against in the LaTeX table
REFERENCE_MODEL_DIR = "bge_large_en_v1.5"

# (column in result files, label in table)
METRICS = (
    ("RFR", "Avg. RFR"),
    ("MRR@10", "MRR@10"),
    ("NDCG@10", "nDCG@10"),
    ("MAP", "MAP"),
)
# Report differences of these metrics in percentage points, as in the benchmark table
PERCENT_METRICS = {"MRR@10", "NDCG@10", "MAP"}

N_BOOTSTRAP = 2_000
SEED = 42


# --- Load per-query scores ---
def load_scores(path):
    df = pd.read_csv(path).sort_values("sample_nr").set_index("sample_nr")
    return df[[metric for metric, _ in METRICS]]


models = {}  # directory_name -> (display name, pooling, per-query scores)
for info in get_all_models().values():
    pooling = info["pooling"]
    path = SCORES_DIR / pooling / f"RerankingEvaluator_{info['directory_name']}_whole_sample_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing per-query results: {path}")
    models[info["directory_name"]] = (info["display_name"], pooling, load_scores(path))

bm25_dir = get_bm25_directory_name()
bm25_path = SCORES_DIR / bm25_dir / f"{bm25_dir}_whole_sample_results.csv"
if not bm25_path.exists():
    raise FileNotFoundError(f"Missing per-query results: {bm25_path}")
models[bm25_dir] = (get_bm25_display_name(), None, load_scores(bm25_path))

if REFERENCE_MODEL_DIR not in models:
    raise KeyError(f"Reference model '{REFERENCE_MODEL_DIR}' not in models_config.yaml")

# Pairing requires identical queries
queries = models[REFERENCE_MODEL_DIR][2].index
for name, (_, _, scores) in models.items():
    if not scores.index.equals(queries):
        raise ValueError(f"Queries of '{name}' differ from those of '{REFERENCE_MODEL_DIR}'")
n_queries = len(queries)


# --- Pairwise tests ---
pairs = list(combinations(models, 2))
rows = []
diffs = []  # one column per (pair, metric), for the bootstrap
for a, b in pairs:
    for metric, _ in METRICS:
        x = models[a][2][metric].to_numpy(dtype=float)
        y = models[b][2][metric].to_numpy(dtype=float)
        scale = 100 if metric in PERCENT_METRICS else 1
        d = (x - y) * scale
        diffs.append(d)
        t_test = stats.ttest_rel(x, y)
        try:  # Zero differences are dropped; fails if all differences are zero
            wilcoxon_p = stats.wilcoxon(x, y).pvalue
        except ValueError:
            wilcoxon_p = np.nan
        rows.append({
            "model_a": a, "pooling_a": models[a][1],
            "model_b": b, "pooling_b": models[b][1],
            "metric": metric,
            "mean_a": x.mean() * scale, "mean_b": y.mean() * scale,
            "diff": d.mean(), "se": d.std(ddof=1) / np.sqrt(n_queries),
            "t_p": t_test.pvalue, "wilcoxon_p": wilcoxon_p,
            "share_ties": np.mean(d == 0),
        })

# Bootstrap: resample queries with the same draws for all comparisons
rng = np.random.default_rng(SEED)
weights = rng.multinomial(n_queries, np.full(n_queries, 1 / n_queries), size=N_BOOTSTRAP)
boot_means = weights @ np.column_stack(diffs) / n_queries
ci = np.percentile(boot_means, [2.5, 97.5], axis=0)

results = pd.DataFrame(rows)
results["boot_ci_low"], results["boot_ci_high"] = ci
results.to_csv(DEST_DIR / "significance_test.csv", index=False)
print(f"Pairwise tests saved to {DEST_DIR / 'significance_test.csv'}")


# --- LaTeX table: each model against the reference ---
def stars(p):
    if p < 0.001:
        return "\\sym{***}"
    if p < 0.01:
        return "\\sym{**}"
    if p < 0.05:
        return "\\sym{*}"
    return ""


def reference_comparison(name, metric):
    """Difference model minus reference, whichever order the pair was stored in."""
    row = results[(results["metric"] == metric)
                  & (((results["model_a"] == name) & (results["model_b"] == REFERENCE_MODEL_DIR))
                     | ((results["model_a"] == REFERENCE_MODEL_DIR) & (results["model_b"] == name)))].iloc[0]
    sign = 1 if row["model_a"] == name else -1
    return sign * row["diff"], row["se"], row["t_p"]


def summarize_to_latex(dest_path):
    reference_label = models[REFERENCE_MODEL_DIR][0]
    reference_pooling = models[REFERENCE_MODEL_DIR][1]
    table = "\\begin{tabular}{l" + "c" * len(METRICS) + "}\n\\toprule\n"
    table += "Model & " + " & ".join(label for _, label in METRICS) + " \\\\\n\\midrule\n"
    for name, (label, pooling, _) in models.items():
        if name == REFERENCE_MODEL_DIR:
            continue
        pooling_label = f" ({pooling.upper() if pooling == 'cls' else pooling.capitalize()})" if pooling else ""
        row = [label + pooling_label]
        for metric, _ in METRICS:
            diff, se, p = reference_comparison(name, metric)
            row.append(f"{diff:.2f}{stars(p)} ({se:.2f})")
        table += " & ".join(row) + " \\\\\n"
    table += "\\midrule\n"
    table += "Observations & " + " & ".join([f"{n_queries:,}"] * len(METRICS)) + " \\\\\n"
    table += "\\bottomrule\n\\end{tabular}\n"
    table += (f"% Notes: Mean per-query difference to {reference_label} ({reference_pooling} pooling); "
              "MRR@10, nDCG@10 and MAP in percentage points. Standard errors of the paired t-test in "
              "parentheses. Significance levels: * p<0.05, ** p<0.01, *** p<0.001.\n")
    with open(dest_path, "w") as f:
        f.write(table)


summarize_to_latex(DEST_DIR / "significance_test.tex")
print(f"Significance test LaTeX table saved to {DEST_DIR / 'significance_test.tex'}")
