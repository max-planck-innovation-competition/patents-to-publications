# ============================================================
# AUTO-GENERATED from models_config.yaml — DO NOT EDIT BY HAND.
# Regenerate with: uv run python generate_makefile_models.py
# ============================================================

# ============
# MODEL LIST
# ============
MODELS = \
	bert-large-uncased \
	scibert_scivocab_uncased \
	patent_sberta \
	specter \
	specter2_base \
	specter2_proximity \
	e5_large_v2 \
	bert_for_patents \
	gte_large \
	bge_large_en_v1.5 \
	patembed_large_IN \
	patembed_large_OUT \
	patembed_large_MIXED \
	pat_specter \
	paecter \
	pubpaecter

# ============
# PRETRAINED MODELS
# ============
download-pretrained-models:
	mkdir -p $(PRETRAINED_DIR)
	@echo "Initializing git-lfs (required to fetch model weights)..."
	git lfs install
	@echo "Downloading pretrained models..."
	@test -d $(PRETRAINED_DIR)/bert-large-uncased || git clone --progress https://huggingface.co/google-bert/bert-large-uncased $(PRETRAINED_DIR)/bert-large-uncased
	@test -d $(PRETRAINED_DIR)/scibert_scivocab_uncased || git clone --progress https://huggingface.co/allenai/scibert_scivocab_uncased $(PRETRAINED_DIR)/scibert_scivocab_uncased
	@test -d $(PRETRAINED_DIR)/patent_sberta || git clone --progress https://huggingface.co/AI-Growth-Lab/PatentSBERTa $(PRETRAINED_DIR)/patent_sberta
	@test -d $(PRETRAINED_DIR)/specter || git clone --progress https://huggingface.co/allenai/specter $(PRETRAINED_DIR)/specter
	@test -d $(PRETRAINED_DIR)/specter2_base || git clone --progress https://huggingface.co/allenai/specter2_base $(PRETRAINED_DIR)/specter2_base
	@test -d $(PRETRAINED_DIR)/specter2_proximity || git clone --progress https://huggingface.co/allenai/specter2_base $(PRETRAINED_DIR)/specter2_proximity
	@test -d $(PRETRAINED_DIR)/e5_large_v2 || git clone --progress https://huggingface.co/intfloat/e5-large-v2 $(PRETRAINED_DIR)/e5_large_v2
	@test -d $(PRETRAINED_DIR)/bert_for_patents || git clone --progress https://huggingface.co/anferico/bert-for-patents $(PRETRAINED_DIR)/bert_for_patents
	@test -d $(PRETRAINED_DIR)/gte_large || git clone --progress https://huggingface.co/thenlper/gte-large $(PRETRAINED_DIR)/gte_large
	@test -d $(PRETRAINED_DIR)/bge_large_en_v1.5 || git clone --progress https://huggingface.co/BAAI/bge-large-en-v1.5 $(PRETRAINED_DIR)/bge_large_en_v1.5
	@test -d $(PRETRAINED_DIR)/patembed_large_IN || git clone --progress https://huggingface.co/datalyes/patembed-large $(PRETRAINED_DIR)/patembed_large_IN
	@test -d $(PRETRAINED_DIR)/patembed_large_OUT || git clone --progress https://huggingface.co/datalyes/patembed-large $(PRETRAINED_DIR)/patembed_large_OUT
	@test -d $(PRETRAINED_DIR)/patembed_large_MIXED || git clone --progress https://huggingface.co/datalyes/patembed-large $(PRETRAINED_DIR)/patembed_large_MIXED
	@test -d $(PRETRAINED_DIR)/pat_specter || git clone --progress https://huggingface.co/mpi-inno-comp/pat_specter $(PRETRAINED_DIR)/pat_specter
	@test -d $(PRETRAINED_DIR)/paecter || git clone --progress https://huggingface.co/mpi-inno-comp/paecter $(PRETRAINED_DIR)/paecter
	@echo "All models from Hugging Face have been downloaded."
