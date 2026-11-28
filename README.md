# Patents to Publications

Benchmarks our embedding models (Pat-SPECTER, PaECTER, PubPaECTER) against baseline models from Hugging Face (BERT, SciBERT, SPECTER/SPECTER2, PatentSBERTa, E5, GTE, BGE, patembed-large, ...) and BM25 on predicting which scientific publications a patent cites.

Each of the 10,000 test cases consists of a randomly drawn EP patent family (query), 5 publications it cites according to [Reliance on Science](https://zenodo.org/records/11461587) (positives), and 25 random publications from around the same time (negatives). Models rank the 30 candidates by cosine similarity of title + abstract embeddings.

Metrics: RFR (rank of first relevant), MAP, MRR@10, nDCG@10.

The workflow has two parts:

1. **Data preparation**, run locally (needs access to PATSTAT on MySQL and to our OpenAlex index).
2. **Benchmark**, run on the MPCDF Raven cluster using Python and SLURM GPU jobs.

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

Parameters (number of positives and negatives, seed) and all paths are set in `config.toml`, which the benchmark also reads: the test data is `target_file` and results go to `output_dir`. Neither can be passed on the command line.

## 2. Setup on MPCDF (once per user)

Log in to Raven:

```sh
ssh -m hmac-sha2-512 ${USERNAME}@gate1.mpcdf.mpg.de
ssh raven
```

To clone via SSH, Raven needs its own SSH key registered with GitHub (once per user). Generate it (Enter accepts the default path, passphrase optional) and print the public key:

```sh
ssh-keygen -t ed25519 -C "${USER}@raven"
cat ~/.ssh/id_ed25519.pub
```

Add it to your personal GitHub profile (not the organisation's settings): profile picture → *Settings → SSH and GPG keys → New SSH key*, choose *Key type: Authentication Key*, and paste the printed line.
The following line confirms that it works:

```sh
ssh -T git@github.com   # "Hi <user>! You've successfully authenticated ..."
```

Then clone the repo:

```sh
cd ~/PROJECTS
git clone git@github.com:max-planck-innovation-competition/patents-to-publications.git
cd patents-to-publications
```

Alternatively, clone via `https://github.com/max-planck-innovation-competition/patents-to-publications.git` and enter a GitHub personal access token (*Settings → Developer settings*) as the password.

Raven only gets what is pushed to GitHub, so commit and push local changes first.

Create the Python environment in `.venv` inside the repo (gitignored):

```sh
module load python-waterboa/2025.06
python3 --version   # 3.13.5
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "torch>=2.3,<3" adapters==1.2.0 pandas==2.2.3 PyYAML==6.0.2 rank-bm25==0.2.2 \
    scikit-learn==1.7.1 sentence-transformers==5.1.0 transformers==4.51.3 tqdm==4.67.1 scipy
```

Download the models into `pretrained_models/` (gitignored), with the environment still active:

```sh
module load git-lfs
make download-pretrained-models   # from Hugging Face
make download-pubpaecter          # PubPaECTER (not on Hugging Face) from MPDL Keeper
```

Models already present are skipped, so after adding a model to `models_config.yaml` the same commands fetch only the new one. If a download was interrupted, delete that model's folder and run them again.

## 3. Benchmark on MPCDF

Upload the test data to MPCDF:

```sh
export USERNAME=your_username
scp -C -o MACs=hmac-sha2-512 \
    -o ProxyJump=${USERNAME}@gate1.mpcdf.mpg.de \
    ./data/patents_to_publications.jsonl \
    ${USERNAME}@raven.mpcdf.mpg.de:~/PROJECTS/patents-to-publications/data/
```

In every new login, load the Python module and activate the environment before running Make. The Makefile passes this Python executable to the SLURM jobs; no container is needed.

```sh
ssh -m hmac-sha2-512 ${USERNAME}@gate1.mpcdf.mpg.de
ssh raven
cd ~/PROJECTS/patents-to-publications
module load python-waterboa/2025.06
source .venv/bin/activate
```

Then run the tests:

```sh
make test          # one SLURM job for BM25 (CPU) and one per model (CLS and mean pooling, GPU)
```

You regularly check the status :
```sh
squeue --me        # done when only the header is left (ST: PD = pending, R = running)
sacct -X -S today --format=JobID,JobName%28,State,Elapsed   # all jobs should be COMPLETED
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
| `output/significance_test.csv` | Paired tests of per-query scores (t-test, Wilcoxon signed-rank, bootstrapped 95% CI) for all pairs of models, each at its default pooling, and BM25 |
| `output/significance_test.tex` | LaTeX table: per-query difference of each model to BGE-large-en-v1.5, with paired t-test standard errors |

## Models

`models_config.yaml` is the single source of truth for the models. Each entry has:

- `huggingface_id`: repository to download from (`null` for PubPaECTER)
- `display_name`: name in tables
- `directory_name`: folder in `pretrained_models/` and key in result file names
- `pooling`: default pooling used for the significance test (the better of CLS and mean)
- optional `normalize` (L2-normalize embeddings) and `q_text`/`doc_text` (prefixes for query and candidate texts, used by patembed)
- `include_in_csv`

The order of entries defines the row order of the tables and the order of job submission. After changing the file, regenerate `models.mk` with `make generate-makefile-models`.
