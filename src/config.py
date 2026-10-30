"""Benchmark settings, read from config.toml."""
import sys
import tomllib
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_PATH / "config.toml"

with open(CONFIG_PATH, "rb") as f:
    SETTINGS = tomllib.load(f)

# Relative paths in config.toml are resolved from the repository root
TEST_DATASET_PATH = BASE_PATH / SETTINGS["directories"]["target_file"]
OUTPUT_PATH = BASE_PATH / SETTINGS["directories"]["output_dir"]


if __name__ == "__main__":
    # The makefile reads paths this way, e.g. python -m src.config directories.target_file
    section, key = sys.argv[1].split(".")
    print((BASE_PATH / SETTINGS[section][key]).as_posix())
