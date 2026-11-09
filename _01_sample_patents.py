"""Sample EP families with at least `n_positive` X/Y scientific NPL citations
from SEA or specific ISA.
"""

import tomllib
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values
from sqlalchemy import create_engine, text, URL
from tqdm import tqdm

dotenv = dotenv_values()
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)

SCHEMA = config["sql"]["schema"]
CACHE_FILE = Path(config["directories"]["ep_xy_npl_cache"])
SAMPLE_FILE = Path(config["directories"]["sample_citations"])
NPL_FILE = Path(config["directories"]["sample_npl"])
MIN_XY_NPL = config["parameters"]["n_positive"]
N_SAMPLE = round(config["parameters"]["n_xy_sample"]
                 * (1 + config["parameters"]["xy_oversample"]))
SEED = config["parameters"]["seed"]


def query_citations(connection):
    """Qualifying NPL citations of EP applications, one row per citation."""
    # The subquery joins all categories of a citation, the EXISTS keeps those with X or
    # Y in any code (LIKE ignores case). NPL ids '' and '0' are placeholders of patent
    # citations
    query = text(f"""
        SELECT t201.appln_id, t201.docdb_family_id, t201.appln_filing_date, t201.granted,
               t211.publn_nr, t211.publn_kind, t212.citn_origin, t212.citn_gener_auth,
               (SELECT GROUP_CONCAT(DISTINCT UPPER(TRIM(c.citn_categ))
                                    ORDER BY UPPER(TRIM(c.citn_categ)) SEPARATOR ',')
                FROM {SCHEMA}.tls215_citn_categ AS c
                WHERE c.pat_publn_id = t212.pat_publn_id
                    AND c.citn_replenished = t212.citn_replenished
                    AND c.citn_id = t212.citn_id) AS citn_categ,
               t214.npl_publn_id, t214.npl_type, t214.npl_biblio, t214.npl_author,
               t214.npl_title1, t214.npl_title2, t214.npl_publn_date, t214.npl_doi
        FROM {SCHEMA}.tls201_appln AS t201
        INNER JOIN {SCHEMA}.tls211_pat_publn AS t211 ON t211.appln_id = t201.appln_id
        INNER JOIN {SCHEMA}.tls212_citation AS t212 ON t212.pat_publn_id = t211.pat_publn_id
        INNER JOIN {SCHEMA}.tls214_npl_publn AS t214
            ON t214.npl_publn_id = t212.cited_npl_publn_id
        WHERE t201.appln_auth = 'EP'
            AND t212.cited_npl_publn_id NOT IN ('', '0')
            AND t214.npl_type IN ('s', 'a', 'c', 'i', 'b')
            AND (t212.citn_origin = 'SEA'
                 OR (t212.citn_origin = 'ISR'
                     AND t212.citn_gener_auth IN ('EP', 'US', 'KR', 'CN', 'JP')))
            AND EXISTS (
                SELECT 1 FROM {SCHEMA}.tls215_citn_categ AS xy
                WHERE xy.pat_publn_id = t212.pat_publn_id
                    AND xy.citn_replenished = t212.citn_replenished
                    AND xy.citn_id = t212.citn_id
                    AND (xy.citn_categ LIKE '%X%' OR xy.citn_categ LIKE '%Y%'));
    """)
    chunks = pd.read_sql(query, connection.execution_options(stream_results=True),
                         chunksize=500_000)
    return pd.concat(list(tqdm(chunks, desc="Reading chunks", unit="chunk")), ignore_index=True)


def get_earliest_ep_applications(connection, family_ids):
    """`appln_id` of the earliest EP application of each given DOCDB family."""
    members = pd.read_sql(text(f"""
        SELECT docdb_family_id, appln_id, appln_filing_date, earliest_publn_date
        FROM {SCHEMA}.tls201_appln
        WHERE appln_auth = 'EP';
    """), connection)
    members = members[members["docdb_family_id"].isin(set(family_ids))]
    # Unknown dates are 9999-12-31 in PATSTAT, so they sort last
    return (members.sort_values(["appln_filing_date", "earliest_publn_date", "appln_id"])
                   .drop_duplicates("docdb_family_id")["appln_id"])


if __name__ == "__main__":
    url = URL.create("mysql+pymysql", username=dotenv['SQL_USER'],
                     password=dotenv['SQL_PASSWORD'], host=config['sql']['host'],
                     port=config['sql']['port'])
    engine = create_engine(url)
    if CACHE_FILE.exists():
        print(f"Reading cached query result from '{CACHE_FILE}'")
        citations = pd.read_parquet(CACHE_FILE)
    else:
        print("Querying X/Y NPL citations of EP applications (takes hours)...")
        with engine.connect() as connection:
            citations = query_citations(connection)
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        citations.to_parquet(CACHE_FILE, index=False)
        print(f"Query result cached in '{CACHE_FILE}'")
    citations["doi"] = (citations.pop("npl_doi").fillna("").str.strip().str.lower()
                        .str.replace(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "",
                                     regex=True))
    print(f"{citations.shape[0]:,} X/Y citations of {citations['npl_publn_id'].nunique():,} NPL "
          f"records by {citations['appln_id'].nunique():,} EP applications "
          f"({citations['docdb_family_id'].nunique():,} DOCDB families)")

    # Query the earliest EP application of each family
    print("Querying the earliest EP application of each family...")
    with engine.connect() as connection:
        earliest = get_earliest_ep_applications(connection, citations["docdb_family_id"])
    citations = citations[citations["appln_id"].isin(earliest)]
    print(f"{citations['docdb_family_id'].nunique():,} families whose earliest EP "
          "application has one")

    # Distinct NPL documents per application, over all its publications
    n_xy = citations.groupby("appln_id")["npl_publn_id"].nunique()
    n_xy = n_xy[n_xy >= MIN_XY_NPL]
    citations = citations[citations["appln_id"].isin(n_xy.index)]
    families = citations["docdb_family_id"].drop_duplicates().sort_values()
    print(f"{families.shape[0]:,} of them cite >= {MIN_XY_NPL} NPL documents")
    if len(families) < N_SAMPLE:
        raise SystemExit(f"Only {families.shape[0]:,} families to sample {N_SAMPLE:,} from")
    sampled_families = families.sample(n=N_SAMPLE, random_state=SEED)
    sample = citations[citations["docdb_family_id"].isin(sampled_families)]

    # Write out
    cit_order = ["appln_id", "docdb_family_id", "appln_filing_date", "granted", "publn_nr",
                "publn_kind", "citn_origin", "citn_categ", "citn_gener_auth", "npl_publn_id"]
    sample = sample.sort_values(["appln_id", "publn_nr", "publn_kind", "citn_origin", "npl_publn_id"])
    sample[cit_order].to_csv(SAMPLE_FILE, index=False)
    npl_order = ["npl_publn_id", "npl_type", "npl_biblio", "npl_author", "npl_title1",
                 "npl_title2", "npl_publn_date", "doi"]
    npl = sample[npl_order].drop_duplicates("npl_publn_id").sort_values("npl_publn_id")
    npl.to_csv(NPL_FILE, index=False)

    # Statistics
    print(f"Sample of {N_SAMPLE:,} families: {len(sample):,} citations written to "
          f"'{SAMPLE_FILE}', {len(npl):,} NPL documents ({(npl['doi'] != '').mean():.1%} "
          f"with DOI) to '{NPL_FILE}'")
    print("NPL documents per application in the sample:")
    print(n_xy[n_xy.index.isin(sample["appln_id"])].describe().round(1).to_string())
    print("Citations by NPL type, origin and authority, and category:")
    for columns in (["npl_type"], ["citn_origin", "citn_gener_auth"], ["citn_categ"]):
        print(sample.groupby(columns).size().sort_values(ascending=False).head(20).to_string())
