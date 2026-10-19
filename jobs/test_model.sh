#!/bin/bash -e
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=92500
#SBATCH --time=5:00:00

MODEL_NAME=$1
TEST_DATA_PATH=$2
PYTHON=$3

cd "$SLURM_SUBMIT_DIR"
PRETRAINED_MODEL_PATH="$SLURM_SUBMIT_DIR/pretrained_models/$MODEL_NAME"
START=$(date +%s)

echo ">>> Running CLS pooling for $MODEL_NAME on $TEST_DATA_PATH"
srun "$PYTHON" -m src.main test-ranking \
    --test-model-path "$PRETRAINED_MODEL_PATH" \
    --test-dataset-path "$TEST_DATA_PATH" \
    --pooling cls

echo ">>> Running MEAN pooling for $MODEL_NAME on $TEST_DATA_PATH"
srun "$PYTHON" -m src.main test-ranking \
    --test-model-path "$PRETRAINED_MODEL_PATH" \
    --test-dataset-path "$TEST_DATA_PATH" \
    --pooling mean

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
