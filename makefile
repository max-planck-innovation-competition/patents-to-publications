# ============
# CONFIGURATION
# ============
BASE_PATH := $(CURDIR)
SINGULARITY_CONTAINER := $(HOME)/CONTAINERS/paecter
# `module load apptainer` is chained here so it runs in the SAME shell as the
# `singularity` call (each recipe line is its own subshell). PYTHONPATH is passed
# via --env so it reaches the container regardless of recipe-line prefixes.
SINGULARITY_EXEC := module load apptainer &&  module load git-lfs && singularity exec --bind $(BASE_PATH) --env PYTHONPATH=$(BASE_PATH) $(SINGULARITY_CONTAINER)

# Python 3.11+ environment with requirements.txt, used for data preparation
PYTHON ?= python

PRETRAINED_DIR := $(BASE_PATH)/pretrained_models
DATA_DIR := $(BASE_PATH)/data
TEST_DATA := $(DATA_DIR)/patents_to_publications.jsonl
JOB_DIR := $(BASE_PATH)/jobs
LOG_DIR := $(BASE_PATH)/logs/test_model
OUTPUT_DIR := $(BASE_PATH)/output

# MODELS list and the download-pretrained-models target are auto-generated
# from models_config.yaml. Regenerate with:
#   make generate-makefile-models
include models.mk

# ============
# CONTAINER
# ============

build-container:
	@echo "Building container..."
	mkdir -p $(dir $(SINGULARITY_CONTAINER))
	cd environment && \
	module load apptainer && \
	apptainer build $(SINGULARITY_CONTAINER) Singularity.paecter

# ============
# DATA PREPARATION (local, needs PATSTAT access, see README)
# ============
prepare-data:
	$(PYTHON) ./01_get_english_openalex.py
	$(PYTHON) ./02_get_ppc.py
	$(PYTHON) ./03_merge_and_analyse.py

# ============
# MODELS
# ============
download-pubpaecter:
	mkdir -p $(PRETRAINED_DIR)
	@test -f $(PRETRAINED_DIR)/pub_paecter.zip || curl -L "https://keeper.mpdl.mpg.de/seafhttp/f/a3476bef4e0944efb205/?op=view" -o $(PRETRAINED_DIR)/pub_paecter.zip
	@# The zip already contains a top-level "pubpaecter/" folder, so extract into
	@# $(PRETRAINED_DIR) directly to get $(PRETRAINED_DIR)/pubpaecter/config.json
	@test -f $(PRETRAINED_DIR)/pubpaecter/config.json || unzip -o $(PRETRAINED_DIR)/pub_paecter.zip -d $(PRETRAINED_DIR)

generate-makefile-models:
	@echo "Generating models.mk from models_config.yaml (requires pyyaml)..."
	PYTHONPATH=$(BASE_PATH) python3 ./generate_makefile_models.py

# ============
# PRE-FLIGHT CHECKS
# ============
check-data:
	@echo "Checking test data..."
	@test -f "$(TEST_DATA)" || { \
		echo "ERROR: Test data not found at: $(TEST_DATA)"; \
		echo "       Run 'make prepare-data' locally and upload the file (see README)."; \
		exit 1; \
	}
	@echo "Test data found: $(TEST_DATA)"

check-models:
	@echo "Checking pretrained models..."
	@missing=0; \
	for model in $(MODELS); do \
		if [ ! -d "$(PRETRAINED_DIR)/$$model" ]; then \
			echo "  MISSING:    $$model (directory not found)"; missing=1; \
		elif [ ! -f "$(PRETRAINED_DIR)/$$model/config.json" ]; then \
			echo "  INCOMPLETE: $$model (no config.json - check git-lfs pull)"; missing=1; \
		fi; \
	done; \
	if [ $$missing -ne 0 ]; then \
		echo "ERROR: Some models are missing or incomplete."; \
		echo "       Run 'make download-pretrained-models download-pubpaecter' (and ensure git-lfs is installed)."; \
		exit 1; \
	fi; \
	echo "All models present."

# ============
# TESTS
# ============
test: check-data check-models
	@echo "Deleting $(OUTPUT_DIR)/ranking"
	rm -rf "$(OUTPUT_DIR)/ranking"

	@echo "Testing BM25"
	$(SINGULARITY_EXEC) python3 ./src/main.py test-bm25 --test-dataset-path "$(TEST_DATA)"

	@echo "Testing all transformer models"
	mkdir -p $(LOG_DIR)
	@for model in $(MODELS); do \
		echo "Submitting $$model"; \
		cd $(JOB_DIR) && sbatch \
			--job-name=test-$$model \
			--output=$(LOG_DIR)/test-$${model}_%A_%a.out \
			test_model.sh $$model $(TEST_DATA); \
	done

# =======================
# RESULTS
# =======================
generate-table:
	$(SINGULARITY_EXEC) python3 ./src/models/generate_benchmark_table.py

test-significance:
	@echo "Conducting statistical significance test"
	$(SINGULARITY_EXEC) python3 ./src/significance_test/on_cross_corpus_data.py

clean-logs:
	rm -rf $(LOG_DIR)

check-logs:
	@echo "Checking logs in $(LOG_DIR)"
	@ls -lh $(LOG_DIR)

	@echo "	Checking for errors..."
	-@grep -i "err" $(LOG_DIR)/*.out

	@echo "	Checking for exceptions..."
	-@grep -i "except" $(LOG_DIR)/*.out

	@echo "	Checking for warnings..."
	-@grep -i "warn" $(LOG_DIR)/*.out

	@echo "Done checking logs"
