#!/bin/bash -e
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=1:00:00

PYTHON=$1

# The test data path is read from config.toml
cd "$SLURM_SUBMIT_DIR"
START=$(date +%s)

echo ">>> Running BM25"
srun "$PYTHON" -m src.main test-bm25

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
