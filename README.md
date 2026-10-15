# Patents to Publications

Benchmarks our embedding models (Pat-SPECTER, PaECTER, PubPaECTER) against baseline models from Hugging Face (BERT, SciBERT, SPECTER/SPECTER2, PatentSBERTa, E5, GTE, BGE, patembed-large, ...) and BM25 on predicting which scientific publications a patent cites.

Each of the 10,000 test cases consists of a randomly drawn EP patent family (query), 5 publications it cites according to [Reliance on Science](https://zenodo.org/records/11461587) (positives), and 25 random publications from around the same time (negatives). Models rank the 30 candidates by cosine similarity of title + abstract embeddings.

Metrics: RFR (rank of first relevant), MAP, MRR@10, nDCG@10.

The workflow has two parts:

1. **Data preparation**, run locally (needs access to PATSTAT on MySQL and to our OpenAlex index).
2. **Benchmark**, run on the MPCDF Raven cluster inside an Apptainer container with SLURM GPU jobs.

All scripts are run from the repository root. Inputs and intermediate files go to `data/` (not committed), all results to `output/`.

## 1. Data preparation

Requirements:

- Python 3.11+ with `pip install -r requirements.txt`
- A `.env` file in the repository root:
  ```
  SQL_USER =
  SQL_PASSWORD =
  ```
- The datasets referenced in `config.toml`: Reliance on Science's patent-to-paper citations (2024-07: `_pcs_oa.csv`; we keep EP patents), the cleaned and replenished OpenAlex index from [publications_index](https://github.com/max-planck-innovation-competition/publications_index), and the PATSTAT MySQL schema.

Run (or `make prepare-data`):

```sh
python 01_get_english_openalex.py   # OpenAlex works with English title + abstract
python 02_get_ppc.py                # Reliance on Science EP patent-paper citations, aggregated to DOCDB families via PATSTAT
python 03_merge_and_analyse.py      # patent texts from PATSTAT, citation-lag analysis, negative sampling
```

| File | Content |
| --- | --- |
| `data/english_oa_ids.txt`, `data/english_oa_text.parquet` | Usable OpenAlex works |
| `data/ppc_families.ftr` | Family-level patent-to-paper citations (families with at least 5 cited papers) |
| `data/patents_to_publications.jsonl` | **Test data**: one line per family with `query`, `pos`, `neg`, each item having `id`, `title`, `abstract` |
| `output/02_missing_patents.log` | Patents not found in PATSTAT |
| `output/kde_citation-lag.pdf` | Distribution of the time lag between citing patent and cited paper |

Parameters (number of positives and negatives, seed) are set in `config.toml`.

Then upload the test data to MPCDF:

```sh
export USERNAME=your_username
scp -C -o MACs=hmac-sha2-512 \
    -o ProxyJump=${USERNAME}@gate1.mpcdf.mpg.de \
    ./data/patents_to_publications.jsonl \
    ${USERNAME}@raven.mpcdf.mpg.de:~/PROJECTS/patents-to-publications/data/
```

## 2. Benchmark on MPCDF

### Setup

```sh
ssh -m hmac-sha2-512 ${USERNAME}@gate1.mpcdf.mpg.de
ssh raven
cd ~/PROJECTS
git clone git@github.com:max-planck-innovation-competition/patents-to-publications.git
cd patents-to-publications
```

The container `~/CONTAINERS/paecter` is the same as in [cross-corpus_testing](https://github.com/max-planck-innovation-competition/cross-corpus_testing). Copy it from there, or build it with `make build-container` (definition: `environment/Singularity.paecter`).

Download the models:

```sh
module load apptainer git-lfs
make download-pretrained-models   # from Hugging Face
make download-pubpaecter          # PubPaECTER (not on Hugging Face) from MPDL Keeper
```

### Run

```sh
make test          # BM25 inline, then one SLURM job per model (CLS and mean pooling)
squeue --me        # wait until all jobs are finished
make check-logs
make generate-table
make test-significance
```

`make test` deletes `output/ranking/` before it starts.

### Results

| Path | Content |
| --- | --- |
| `output/ranking/{cls,mean,bm25}/` | Per model: aggregate scores (`..._results.csv`) and per-query scores (`..._whole_sample_results.csv`) |
| `output/benchmark_results_table.tex` | LaTeX table, CLS and mean pooling (without BM25) |
| `output/benchmark_results_table_cross_corpus.csv` | Same table as CSV, consumed by [lm-vue-client](https://github.com/max-planck-innovation-competition/lm-vue-client) |
| `output/patspecter_significance_test.tex` | OLS regressions of per-query scores on model dummies, Pat-SPECTER as reference |

## Models

`models_config.yaml` is the single source of truth for the models. Each entry has:

- `huggingface_id`: repository to download from (`null` for PubPaECTER)
- `display_name`: name in tables
- `directory_name`: folder in `pretrained_models/` and key in result file names
- `pooling`: default pooling used for the significance test
- optional `normalize` (L2-normalize embeddings) and `q_text`/`doc_text` (prefixes for query and candidate texts, used by patembed)
- `include_in_csv`

The order of entries defines the row order of the tables and the order of job submission. After changing the file, regenerate `models.mk` with `make generate-makefile-models`.
