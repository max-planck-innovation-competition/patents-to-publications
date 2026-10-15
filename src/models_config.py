"""Utility to load and access model configuration from models_config.yaml"""
import yaml
from pathlib import Path
from typing import Dict, List, Optional


# Path to the config file (relative to project root)
CONFIG_PATH = Path(__file__).parent.parent / "models_config.yaml"


def load_config() -> Dict:
    """Load the model configuration from YAML file."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def get_all_models() -> Dict:
    """Get all transformer models configuration."""
    config = load_config()
    return config.get('models', {})


def get_model_info(model_key: str) -> Optional[Dict]:
    """Get configuration for a specific model by its key."""
    models = get_all_models()
    return models.get(model_key)


def get_model_directory_names() -> List[str]:
    """Get list of all model directory names (for makefile MODELS variable)."""
    models = get_all_models()
    return [info['directory_name'] for info in models.values()]


def get_model_display_names() -> Dict[str, str]:
    """Get mapping from directory_name to display_name."""
    models = get_all_models()
    return {info['directory_name']: info['display_name'] for info in models.values()}


def get_model_huggingface_ids() -> Dict[str, str]:
    """Get mapping from directory_name to huggingface_id."""
    models = get_all_models()
    return {info['directory_name']: info['huggingface_id'] for info in models.values()}


def get_model_pooling_methods() -> Dict[str, str]:
    """Get mapping from directory_name to default pooling method."""
    models = get_all_models()
    return {info['directory_name']: info['pooling'] for info in models.values()}


def get_model_order() -> List[str]:
    """Get the order of models for benchmark tables (by directory_name).

    The order is derived directly from the order of entries in
    models_config.yaml, keeping the YAML the single source of truth.
    """
    return get_model_directory_names()


def get_display_name_for_directory(directory_name: str) -> str:
    """Return the display_name for a given directory_name.

    Raises KeyError if the directory_name is not present in the config, which
    guards against silently passing an unknown reference category to the
    significance-test regressions.
    """
    mapping = get_model_display_names()
    return mapping[directory_name]


def get_bm25_display_name() -> str:
    """Get BM25 display name."""
    config = load_config()
    return config.get('bm25', {}).get('display_name', 'BM25')


def get_bm25_directory_name() -> str:
    """Get BM25 directory name."""
    config = load_config()
    return config.get('bm25', {}).get('directory_name', 'bm25')


def get_model_q_text(directory_name: str) -> Optional[str]:
    """Get query text prefix for a given directory_name."""
    models = get_all_models()
    for info in models.values():
        if info['directory_name'] == directory_name:
            return info.get('q_text')
    return None


def get_model_doc_text(directory_name: str) -> Optional[str]:
    """Get document text prefix for a given directory_name (used for both positive and negative samples)."""
    models = get_all_models()
    for info in models.values():
        if info['directory_name'] == directory_name:
            return info.get('doc_text')
    return None


def get_model_normalize(directory_name: str) -> bool:
    """Get normalize setting for a given directory_name."""
    models = get_all_models()
    for info in models.values():
        if info['directory_name'] == directory_name:
            return info.get('normalize', False)
    return False
