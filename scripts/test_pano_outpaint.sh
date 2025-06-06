#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/.."

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# Change these two lines to point at your actual image and experiment name:
INPUT_IMAGE="${PROJECT_DIR}/inputs/bed1.jpg"
EXP_NAME="one_shot_outpaint"

# You can leave these alone (they control GPUs / batch size etc for dataset—but will be ignored
# when --input_image is used).
n_nodes=1
n_gpus_per_node=1
torch_num_workers=0
batch_size=8
exp_name="${EXP_NAME}"

# Call test.py in “single‐image” mode by passing --input_image
CUDA_VISIBLE_DEVICES='0' python -u "${PROJECT_DIR}/test.py" \
    configs/pano_generation_outpaint.yaml \
    --ckpt_path="${PROJECT_DIR}/weights/pano_outpaint.ckpt" \
    --exp_name="${exp_name}" \
    --input_image="${INPUT_IMAGE}"