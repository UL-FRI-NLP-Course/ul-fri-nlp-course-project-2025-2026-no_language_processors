#!/bin/bash
#SBATCH --job-name=vllm_project
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/vllm_%j.out

PORT=$(shuf -i 8800-9500 -n 1)
NODE=$(hostname)

# Define paths
PROJ_DIR="/d/hpc/projects/onj_fri/no-language-processors-v2"
VLLM_IMAGE="$PROJ_DIR/vllm.sif"

echo "---------------------------------------------"
echo "vLLM Jupyter running on node: $NODE"
echo "Port: $PORT"
echo "Project Home: $PROJ_DIR"
echo "SSH tunnel command (run on your laptop):"
echo "  ssh -N -L ${PORT}:${NODE}:${PORT} gm64359@hpc-login.arnes.si"
echo "Then open: http://localhost:${PORT}"
echo "---------------------------------------------"

# Keep cache in the project folder to save your $HOME space
export HF_HOME="$PROJ_DIR/hf_cache"
mkdir -p $HF_HOME

apptainer exec --nv \
    --bind /d/hpc/projects/onj_fri:/d/hpc/projects/onj_fri \
    --bind $HOME:$HOME \
    $VLLM_IMAGE \
    jupyter lab \
        --no-browser \
        --port=$PORT \
        --ip=0.0.0.0 \
        --notebook-dir=$PROJ_DIR