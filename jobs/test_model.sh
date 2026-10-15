#!/bin/bash -e
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=92500
#SBATCH --time=1:00:00

MODEL_NAME=$1
TEST_DATA_PATH=$2

module load apptainer
# No host CUDA module needed: the container's torch ships its own CUDA 11.7
# runtime (see environment/Singularity.paecter) and --nv injects the host
# NVIDIA driver. The driver only needs to support CUDA 11.7 (driver >= 515).
export BASE=$(cd .. && pwd)
export APPTAINERENV_PREPEND_PATH=$HOME/.local/bin
# The container is using the host's PyPI version (2.2.2 from ~/.local/lib/python3.8/site-packages), not the git-installed fork. The container's python is finding the host's user site-packages first.
# The fix: disable user site-packages in the container so it only uses the container's system-installed fork. 
export APPTAINERENV_PYTHONNOUSERSITE=1
export PRETRAINED_MODEL_PATH=$BASE/pretrained_models/$MODEL_NAME
CONTAINER=$HOME/CONTAINERS/paecter

APPTAINER_ARGS=" \
    apptainer exec --nv --bind $BASE $CONTAINER \
"

cd $BASE
START=$(date +%s)

echo ">>> Running CLS pooling for $MODEL_NAME on $TEST_DATA_PATH"
srun $APPTAINER_ARGS python3 ./src/main.py test-ranking \
    --test-model-path "$PRETRAINED_MODEL_PATH" \
    --test-dataset-path "$TEST_DATA_PATH" \
    --pooling cls

echo ">>> Running MEAN pooling for $MODEL_NAME on $TEST_DATA_PATH"
srun $APPTAINER_ARGS python3 ./src/main.py test-ranking \
    --test-model-path "$PRETRAINED_MODEL_PATH" \
    --test-dataset-path "$TEST_DATA_PATH" \
    --pooling mean

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
