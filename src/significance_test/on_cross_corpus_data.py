"""
Runs OLS regressions on patent-to-science citation (cross-corpus) performance scores, 
comparing each model to Pat-SPECTER. Outputs a LaTeX table of coefficients, 
standard errors, and significance levels for RFR, MAP, and MRR@10.
"""
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from src.models_config import get_all_models, get_display_name_for_directory


# --- Config ---
SCORES_DIR = Path("./output/ranking")
DEST_DIR = Path("./output")
DEST_DIR.mkdir(parents=True, exist_ok=True)

# Focal model the baselines are compared against (regression base category).
# Derived from the config so the display name can never drift out of sync.
REFERENCE_MODEL_DIR = "pat_specter"
REFERENCE_MODEL = get_display_name_for_directory(REFERENCE_MODEL_DIR)

# --- Model mapping ---
models = get_all_models()

MODEL_SPECS = []

# Add transformer models with their pooling methods
for model_key, model_info in models.items():
    directory_name = model_info['directory_name']
    display_name = model_info['display_name']
    pooling = model_info['pooling']
    MODEL_SPECS.append(
        (f"{pooling}/RerankingEvaluator_{directory_name}_whole_sample_results.csv", display_name, directory_name)
    )

# --- Load and label all data ---
dfs = []
for idx, (rel_path, model_label, model_key) in enumerate(MODEL_SPECS, start=1):
    full_path = SCORES_DIR / rel_path
    if not full_path.exists():
        print(f"File not found: {full_path}")
        continue
    df = pd.read_csv(full_path)
    df["model_name"] = model_label
    df["model_code"] = idx
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# --- Rename MRR@10 with MRR_10
df_all = df_all.rename(columns={'MRR@10': 'MRR_10', 'NDCG@10': 'NDCG_10'})

# --- Set the focal model as base category ---
model_display_names = [spec[1] for spec in MODEL_SPECS]
df_all["model_name"] = pd.Categorical(df_all["model_name"], categories=model_display_names, ordered=True)

# --- Regressions ---
_term = f"C(model_name, Treatment(reference='{REFERENCE_MODEL}'))"
formulas = {
    "Avg. RFR": f"RFR ~ {_term}",
    "MAP": f"MAP ~ {_term}",
    "MRR@10": f"MRR_10 ~ {_term}",
    "NDCG@10": f"NDCG_10 ~ {_term}",
}

results = {}
for label, formula in formulas.items():
    model = smf.ols(formula, data=df_all).fit()
    results[label] = model

# --- LaTeX output ---
def summarize_to_latex(models, dest_path):
    table = "\\begin{table}[htbp]\n\\centering\n\\caption{Statistical Significance Testing of Rank-Aware Evaluation of Pat-SPECTER versus Different Models on a Cross-Corpus Dataset}\n\\begin{tabular}{lcccc}\n\\toprule\n"
    table += " & Avg. RFR & MAP & MRR@10 & NDCG@10 \\\\\n\\midrule\n"

    # First: all coefficients except intercept
    coeff_names = [name for name in models["Avg. RFR"].params.index if name != "Intercept"]
    prefix = f"C(model_name, Treatment(reference='{REFERENCE_MODEL}'))[T."
    for name in coeff_names:
        pretty_name = name.replace(prefix, "").replace("]", "")
        row = [pretty_name]
        for label in ["Avg. RFR", "MAP", "MRR@10", "NDCG@10"]:
            coef = models[label].params.get(name, 0)
            stderr = models[label].bse.get(name, 0)
            pval = models[label].pvalues.get(name, 1)
            if pval < 0.001:
                stars = "\\sym{***}"
            elif pval < 0.01:
                stars = "\\sym{**}"
            elif pval < 0.05:
                stars = "\\sym{*}"
            else:
                stars = ""
            row.append(f"{coef:.3f}{stars} ({stderr:.3f})")
        table += " & ".join(row) + " \\\\\n"

    # Then: intercept (constant)
    const_row = ["Constant"]
    for label in ["Avg. RFR", "MAP", "MRR@10", "NDCG@10"]:
        coef = models[label].params["Intercept"]
        stderr = models[label].bse["Intercept"]
        pval = models[label].pvalues["Intercept"]
        if pval < 0.001:
            stars = "\\sym{***}"
        elif pval < 0.01:
            stars = "\\sym{**}"
        elif pval < 0.05:
            stars = "\\sym{*}"
        else:
            stars = ""
        const_row.append(f"{coef:.3f}{stars} ({stderr:.3f})")
    table += " & ".join(const_row) + " \\\\\n"

    # Add Adjusted R^2 and Observations
    table += "\\midrule\n"
    table += "Adjusted $R^2$"
    for label in ["Avg. RFR", "MAP", "MRR@10", "NDCG@10"]:
        table += f" & {models[label].rsquared_adj:.3f}"
    table += " \\\\\n"

    table += "Observations"
    for label in ["Avg. RFR", "MAP", "MRR@10", "NDCG@10"]:
        table += f" & {int(models[label].nobs):,}"
    table += " \\\\\n"

    table += "\\bottomrule\n\\end{tabular}\n\\begin{flushleft}\\small\n"
    table += "\\textbf{Notes:} Standard errors in parentheses. Base level model is Pat-SPECTER. All models use their best pooling method. "
    table += "Significance levels: $^{*} p<0.05$, $^{**} p<0.01$, $^{***} p<0.001$.\n"
    table += "\\end{flushleft}\n\\end{table}\n"

    with open(dest_path, "w") as f:
        f.write(table)

summarize_to_latex(results, DEST_DIR / "patspecter_significance_test.tex")
print(f"Significance test LaTeX table saved to {DEST_DIR / 'patspecter_significance_test.tex'}")
