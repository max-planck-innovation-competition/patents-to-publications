#!/usr/bin/env python3
"""
Generate ``models.mk`` from ``models_config.yaml``.

``models.mk`` defines the ``MODELS`` variable and the
``download-pretrained-models`` target. The main ``makefile`` simply does
``include models.mk``, so the model list never needs to be hand-edited or
copy-pasted: re-run this script whenever ``models_config.yaml`` changes.

Usage:
    uv run python generate_makefile_models.py
"""
from pathlib import Path

from src.models_config import get_model_directory_names, get_model_huggingface_ids

OUTPUT_FILE = Path(__file__).parent / "models.mk"


def build_models_mk() -> str:
    """Build the contents of models.mk from the central YAML config."""
    directory_names = get_model_directory_names()
    huggingface_ids = get_model_huggingface_ids()

    lines = [
        "# ============================================================",
        "# AUTO-GENERATED from models_config.yaml — DO NOT EDIT BY HAND.",
        "# Regenerate with: uv run python generate_makefile_models.py",
        "# ============================================================",
        "",
        "# ============",
        "# MODEL LIST",
        "# ============",
        "MODELS = \\",
    ]
    for i, dir_name in enumerate(directory_names):
        suffix = "" if i == len(directory_names) - 1 else " \\"
        lines.append(f"\t{dir_name}{suffix}")

    lines += [
        "",
        "# ============",
        "# PRETRAINED MODELS",
        "# ============",
        "download-pretrained-models:",
        "\tmkdir -p $(PRETRAINED_DIR)",
        '\t@echo "Initializing git-lfs (required to fetch model weights)..."',
        "\tgit lfs install",
        '\t@echo "Downloading pretrained models..."',
    ]
    for dir_name, hf_id in zip(directory_names, huggingface_ids.values()):
        # Skip models without a HuggingFace id (e.g. manually-uploaded PubPaECTER)
        if hf_id:
            lines.append(
                f"\t@test -d $(PRETRAINED_DIR)/{dir_name} || "
                f"git clone --progress https://huggingface.co/{hf_id} $(PRETRAINED_DIR)/{dir_name}"
            )
    lines.append('\t@echo "All models from Hugging Face have been downloaded."')

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUTPUT_FILE.write_text(build_models_mk())
    print(f"Wrote {OUTPUT_FILE}")
