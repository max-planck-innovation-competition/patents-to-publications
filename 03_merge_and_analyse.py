"""Merge suitable PPCs with OpenAlex data on family level."""

import tomllib
from pathlib import Path

from dotenv import dotenv_values
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from orjson import dumps
from sqlalchemy import create_engine, text, URL
from tqdm import tqdm

dotenv = dotenv_values()

sns.set_style('whitegrid')
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)

N_POS = config["parameters"]["n_positive"]
N_NEG = config["parameters"]["n_negative"]
N_QUERIES = config["parameters"]["n_queries"]
SEED = config["parameters"]["seed"]

PATENTS_FILE = Path(config["directories"]["ppc_families"])
PAPERS_FILE = Path(config["directories"]["english_oa_text"])
TARGET_FILE = Path(config["directories"]["target_file"])
OUTPUT_FOLDER = Path(config["directories"]["output_dir"])


def get_english_texts(families, connection, table_name, offices=("EP", "WO", "US", "GB")):
    """Find an English title and abstract among the members of each family.

    Our PATSTAT tables have no indexes, so each table is joined separately to a
    temporary table, which costs one full scan per table.
    """
    schema, temp = config['sql']['schema'], f"{dotenv['SQL_USER']}.{table_name}"
    pd.DataFrame({"docdb_family_id": families}).to_sql(
        table_name, connection, schema=dotenv['SQL_USER'], if_exists='replace', index=False)
    members = pd.read_sql(f"""
        SELECT t201.docdb_family_id, t201.appln_id, t201.appln_auth
        FROM {temp} AS temp
        INNER JOIN {schema}.tls201_appln AS t201 ON temp.docdb_family_id = t201.docdb_family_id;
    """, connection)
    members[["appln_id"]].to_sql(table_name, connection, schema=dotenv['SQL_USER'],
                                 if_exists='replace', index=False)
    titles = pd.read_sql(f"""
        SELECT t202.appln_id, t202.appln_title AS title
        FROM {temp} AS temp
        INNER JOIN {schema}.tls202_appln_title AS t202 ON temp.appln_id = t202.appln_id
        WHERE t202.appln_title_lg = 'en';
    """, connection)
    abstracts = pd.read_sql(f"""
        SELECT t203.appln_id, t203.appln_abstract AS abstract
        FROM {temp} AS temp
        INNER JOIN {schema}.tls203_appln_abstr AS t203 ON temp.appln_id = t203.appln_id
        WHERE t203.appln_abstract_lg = 'en' AND t203.appln_abstract <> '';
    """, connection)
    connection.execute(text(f"DROP TABLE {temp}"))
    # Prefer members with English title and abstract, then by office
    members = members.merge(titles, on="appln_id", how="left").merge(abstracts, on="appln_id", how="left")
    members["missing"] = members[["title", "abstract"]].isna().sum(axis=1)
    members["office"] = members["appln_auth"].map({o: i for i, o in enumerate(offices)}).fillna(len(offices))
    members = members.sort_values(["missing", "office", "appln_id"])
    return pd.DataFrame({
        col: members.dropna(subset=col).drop_duplicates("docdb_family_id").set_index("docdb_family_id")[col]
        for col in ("title", "abstract")})


def make_kde_plot(ppc, fname, figsize=(10, 6), percentiles=(1, 5, 50, 95, 99)):
    """Create a KDE plot for the citation lag (delta) with percentiles."""
    # Initialize plot
    plt.figure(figsize=figsize)
    sns.kdeplot(ppc["delta"], fill=True, color="skyblue")
    # Vertical lines for percentiles
    values = np.percentile(ppc["delta"], percentiles)
    ylim = plt.ylim()
    for perc, val in zip(percentiles, values):
        plt.axvline(val, linestyle="--")
        plt.text(val, ylim[1] * 0.95, f"{perc}%", rotation=90,
                 va="top", ha="right", fontsize=9, color="grey")
    # Aesthetics
    plt.xlabel("Time lag between publication of citing patent and cited paper")
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')


if __name__ == '__main__':
    # Read in patents data
    print("Querying texts of patents...")
    ppc = pd.read_feather(PATENTS_FILE).drop(columns="patent")
    families = ppc["docdb_family_id"].drop_duplicates()
    queries = families.sample(min(N_QUERIES, len(families)), random_state=SEED)
    print(f"Drawing {len(queries):,} of {len(families):,} families as queries")
    url = URL.create("mysql+pymysql", username=dotenv['SQL_USER'],
                     password=dotenv['SQL_PASSWORD'], host=config['sql']['host'],
                     port=config['sql']['port'])
    engine = create_engine(url)
    patents = (ppc.loc[ppc["docdb_family_id"].isin(queries), ["appln_id", "docdb_family_id"]]
                  .drop_duplicates(subset="docdb_family_id"))
    table_name = 'temp'
    with engine.connect() as connection:
        patents.to_sql(table_name, connection, schema=dotenv['SQL_USER'],
                       if_exists='replace', index=False)
        query = f"""
        SELECT
            temp.docdb_family_id,
            t202.appln_title AS title,
            t202.appln_title_lg AS title_lg,
            t203.appln_abstract AS abstract,
            t203.appln_abstract_lg AS abstract_lg
        FROM {dotenv['SQL_USER']}.{table_name} AS temp
        LEFT JOIN {config['sql']['schema']}.tls202_appln_title AS t202
            ON temp.appln_id = t202.appln_id
        LEFT JOIN {config['sql']['schema']}.tls203_appln_abstr AS t203
            ON temp.appln_id = t203.appln_id;
        """
        texts = pd.read_sql(query, connection, index_col="docdb_family_id")
        connection.execute(text(f"DROP TABLE {dotenv['SQL_USER']}.{table_name}"))

        # Replace non-English or missing texts with those of another family member
        foreign = texts.index[(texts["title_lg"] != "en") | (texts["abstract_lg"] != "en")]
        print(f"Looking up English texts in other family members for {len(foreign):,} families...")
        alternatives = get_english_texts(foreign, connection, table_name).reindex(texts.index)
    for col in ("title", "abstract"):
        replace = (texts[f"{col}_lg"] != "en") & alternatives[col].notna()
        texts.loc[replace, col] = alternatives.loc[replace, col]
        print(f"Replaced {replace.sum():,} {col}s, {(texts[f'{col}_lg'] != 'en').sum() - replace.sum():,} "
              f"remain without English {col}")
    texts = texts[["title", "abstract"]].to_dict(orient="index")
    del patents

    # Merging in OpenAlex data
    print("Merging publications from OpenAlex...")
    openalex = pd.read_parquet(PAPERS_FILE)
    ppc = ppc.join(openalex, on="oaid", how="left")
    assert ppc["oaid"].isna().sum() == 0
    rename = {"publication_date": "paper_year", "earliest_publn_year": "family_year"}
    ppc = ppc.rename(columns=rename)

    # Compute citation lags
    ppc["delta"] = ppc["paper_year"] - ppc["family_year"]
    dist = ppc["delta"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    print("Distribution of citation lags (in years):")
    print(dist)
    fname = OUTPUT_FOLDER / "kde_citation-lag.pdf"
    make_kde_plot(ppc, fname)
    ppc = ppc[ppc["docdb_family_id"].isin(queries)].drop(columns=["delta", "paper_year"])

    # Set maximum publication ranges
    max_lag = int(dist["1%"])
    min_lag = int(dist["99%"])
    openalex = (openalex.reset_index().set_index("publication_date")
                         .sort_index())
    min_year = ppc["family_year"].min() + max_lag
    openalex = openalex.loc[min_year:]
    print(f"Using up to {openalex.shape[0]:,} English works with abstract published since {min_year}")

    # Write out training records
    ppc = (ppc.sample(frac=1.0, random_state=SEED)
              .sort_values(["family_year", "docdb_family_id"], kind="mergesort")
              .rename(columns={"oaid": "id"}))
    total = ppc["docdb_family_id"].nunique()
    rng = np.random.default_rng(SEED)
    with open(TARGET_FILE, "wb") as f:
        with tqdm(total=total, desc="Generating triplets") as pbar:
            for pub_year, subset in ppc.groupby("family_year"):
                risk_set = openalex.loc[pub_year + max_lag: pub_year + min_lag]
                for patent, df in subset.groupby("docdb_family_id", sort=False, observed=True):
                    foc = texts[patent]
                    pos = df.head(N_POS)[["id", "title", "abstract"]]
                    # Draw positions directly from the risk set, then filter out those that are cited
                    cited = df["id"].unique()
                    positions = rng.choice(risk_set.shape[0], N_NEG + len(cited), replace=False)
                    neg = risk_set.iloc[positions]
                    neg = neg[~neg["id"].isin(cited)].head(N_NEG)
                    record = {
                        "query": {"id": patent, "title": foc["title"], "abstract": foc["abstract"]},
                        "pos": pos.to_dict(orient="records"),
                        "neg": neg.to_dict(orient="records"),
                    }
                    f.write(dumps(record) + b"\n")
                    pbar.update(1)
                openalex = openalex.loc[pub_year + max_lag:]  # Drop old papers
