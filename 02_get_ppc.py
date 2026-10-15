"""Merge suitable PPCs with OpenAlex data on family level."""

import tomllib
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values
from sqlalchemy import create_engine, text, URL
from tqdm import tqdm

dotenv = dotenv_values()
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)

OA_PPC_FILE = Path(config["datasets"]["pcs_oa"])
ENGLISH_OA_IDS = Path(config["directories"]["english_oa_ids"])
MAINTENANCE_FILE = Path(config["directories"]["output_dir"]) / "02_missing_patents.log"
TARGET_FILE = Path(config["directories"]["ppc_families"])
MAINTENANCE_FILE.parent.mkdir(parents=True, exist_ok=True)

MIN_SAMPLE = config["parameters"]["n_positive"]
SEED = config["parameters"]["seed"]
PPC_READ_CHUNK_SIZE = 1_000_000

tqdm.pandas()


def read_ep_ppcs(fpath, chunk_size=PPC_READ_CHUNK_SIZE):
    """Read high-confidence PPCs of EPO patents chunk-wise."""
    dtypes = {"patent": str, "oaid": "uint64", "confscore": "int8"}
    chunks = pd.read_csv(fpath, usecols=dtypes.keys(), dtype=dtypes,
                         chunksize=chunk_size, memory_map=True)
    subsets = [chunk[(chunk["confscore"] == 10) & chunk["patent"].str.startswith("ep")]
               for chunk in tqdm(chunks, desc="Reading PPC chunks", unit="chunk")]
    return pd.concat(subsets, ignore_index=True).drop(columns="confscore")


if __name__ == "__main__":
    print("Reading in PPCs involving EPO patents and complete OA papers")
    ppc = read_ep_ppcs(OA_PPC_FILE)
    print(f"{ppc.shape[0]:,} high-confidence citations by {ppc['patent'].nunique():,} EP patents")
    if ppc.empty:
        raise SystemExit(f"No high-confidence citations by EP patents in '{OA_PPC_FILE}'")
    with open(ENGLISH_OA_IDS, "r") as f:
        oa_ids = {int(line.strip()) for line in f}
    ppc = ppc[ppc["oaid"].isin(oa_ids)]

    # Drop patents w/o sufficient NPL
    ppc["publno"] = ppc["patent"].str.split("-").str[1]
    counts = ppc["publno"].value_counts()
    counts = counts[counts >= MIN_SAMPLE]
    ppc = ppc[ppc["publno"].isin(counts.index)].drop(columns="publno")
    # del counts
    ppc["patent"] = ppc["patent"].str.upper()
    if ppc.empty:
        raise SystemExit(f"No EP patent cites >= {MIN_SAMPLE} papers listed in '{ENGLISH_OA_IDS}'")

    # Get family IDs and publication years
    print(f"Querying families for {ppc['patent'].nunique():,} patents...")
    url = URL.create("mysql+pymysql", username=dotenv['SQL_USER'],
                     password=dotenv['SQL_PASSWORD'], host=config['sql']['host'],
                     port=config['sql']['port'])
    engine = create_engine(url)
    # Match on office and number only: PATSTAT lacks many cited kinds (e.g. old EP A3/A4
    # search reports), but all kinds of one EP number belong to the same application
    ppc["publn"] = ppc["patent"].str.rsplit("-", n=1).str[0]
    patents = ppc["publn"].drop_duplicates().str.split("-", expand=True)
    patents.columns = ["publn_auth", "publn_nr"]
    table_name = 'temp'
    with engine.connect() as connection:
        patents.to_sql(table_name, connection, schema=dotenv['SQL_USER'],
                       if_exists='replace', index=False)
        query = f"""
            SELECT DISTINCT
                temp.*,
                tls211.appln_id,
                tls201.earliest_publn_year,
                tls201.docdb_family_id
            FROM {dotenv['SQL_USER']}.{table_name} AS temp
            INNER JOIN {config['sql']['schema']}.tls211_pat_publn AS tls211
                ON temp.publn_auth = tls211.publn_auth
                AND temp.publn_nr = tls211.publn_nr
            LEFT JOIN {config['sql']['schema']}.tls201_appln AS tls201
                ON tls211.appln_id = tls201.appln_id;
        """
        families = pd.read_sql(query, connection)
        connection.execute(text(f"DROP TABLE {dotenv['SQL_USER']}.{table_name}"))
    families.index = families["publn_auth"] + "-" + families["publn_nr"]
    families = families.drop(columns=["publn_auth", "publn_nr"])
    ambiguous = families.index.duplicated(keep=False)
    if ambiguous.any():
        print(f"Dropping {families.index[ambiguous].nunique():,} numbers with several applications")
        families = families[~ambiguous]

    # Extract dates
    families["earliest_publn_year"] = (families.groupby("docdb_family_id")["earliest_publn_year"]
                                               .transform("min"))

    # Deduplicate on family level
    ppc = ppc.join(families, how="left", on="publn").drop(columns="publn")
    del families
    missing = ppc[ppc["docdb_family_id"].isna()][["patent"]].drop_duplicates()
    missing.to_csv(MAINTENANCE_FILE, index=False)
    ppc = ppc.dropna(subset=["docdb_family_id"])
    ppc["earliest_publn_year"] = ppc["earliest_publn_year"].astype("int16")
    ppc = (ppc.sort_values("earliest_publn_year")
              .drop_duplicates(subset=["docdb_family_id", "oaid"]))

    # Drop patents w/o sufficient NPL on family level
    counts = ppc["docdb_family_id"].value_counts()
    counts = counts[counts >= MIN_SAMPLE]
    ppc = ppc[ppc["docdb_family_id"].isin(counts.index)]
    del counts
    print(f"Number of patents with >= {MIN_SAMPLE} NPLs: {ppc['docdb_family_id'].nunique():,}")

    # Write out
    for c in ["docdb_family_id", "appln_id"]:
        ppc[c] = ppc[c].astype("uint32")
    ppc.to_feather(TARGET_FILE)
    print(f"{ppc.shape[0]:,} citation links between "
          f"{ppc['docdb_family_id'].nunique():,} families and {ppc['oaid'].nunique():,} papers")
