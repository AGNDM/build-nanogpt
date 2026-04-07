#!/bin/bash
#SBATCH --job-name=nanogpt-train-2n8g
#SBATCH --account=a131
#SBATCH --time=08:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=log_3_epoch/%x-%j.out
#SBATCH --error=log_3_epoch/%x-%j.err

cd /iopsstor/scratch/cscs/tong/share/xianrong_liu/build-nanogpt

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

srun --uenv=pytorch/v2.8.0:v1 --view=default bash -c " \
    source /iopsstor/scratch/cscs/tong/share/xianrong_liu/.venv/bin/activate && \
    source /iopsstor/scratch/cscs/tong/share/xianrong_liu/build-nanogpt/cache_env_setup.sh && \
    python -m torch.distributed.run \
        --nnodes=2 \
        --nproc_per_node=4 \
        --node_rank=\$SLURM_NODEID \
        --rdzv_id=\$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \
        train_gpt2_multi_epoch.py \
"
