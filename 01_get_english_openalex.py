"""Get suitable OpenAlex works (English, abstract and title exist)."""

import multiprocessing
import tomllib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from tqdm import tqdm

num_cores = max(1, multiprocessing.cpu_count() // 2)
with open("./config.toml", "rb") as f:
    config = tomllib.load(f)

OA_DIR = Path(config["datasets"]["oa_english_repl"])
TARGET_IDS = Path(config["directories"]["english_oa_ids"])
TARGET_TEXT = Path(config["directories"]["english_oa_text"])
TARGET_IDS.parent.mkdir(parents=True, exist_ok=True)

TYPES = {"article", "book", "preprint", "book-chapter", "report", "dissertation"}


def read_oa_file(fpath):
    """Read a single OpenAlex parquet file and filter for publications."""
    cols = ["id", "title", "abstract", "publication_date", "type"]
    df = pd.read_parquet(fpath, columns=cols)
    df = df[df["type"].isin(TYPES)].drop(columns="type")
    df = df.dropna(how="any")
    df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.year.astype("int16")
    return df


if __name__ == '__main__':
    # Read in
    fpaths = list(OA_DIR.glob("*.parquet"))
    all_chunks = Parallel(n_jobs=num_cores)(
        delayed(read_oa_file)(fpath) for fpath in tqdm(fpaths, desc='Reading OA files')
    )
    df = pd.concat(all_chunks)
    del all_chunks
    df = df.dropna(subset=["abstract", "title"], how="any")
    assert df.isna().sum().sum() == 0

    # Write out list of English OA IDs
    df.index.to_series().to_csv(TARGET_IDS, index=False, header=False)

    # Write out merge filed
    print(f"Writing out {df.shape[0]:,} OA works with English text to '{TARGET_TEXT}'")
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=True),
        TARGET_TEXT,
        row_group_size=250_000,
        write_statistics=False,
        use_dictionary=False
    )
