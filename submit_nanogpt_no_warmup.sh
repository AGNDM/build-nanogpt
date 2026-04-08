#!/bin/bash
#SBATCH --job-name=nanogpt-nowarmup
#SBATCH --account=a131
#SBATCH --time=03:00:00              
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=1          
#SBATCH --gpus-per-node=4            
#SBATCH --cpus-per-task=288
#SBATCH --output=log_no_warmup/%x-%j.out           
#SBATCH --error=log_no_warmup/%x-%j.err            

cd /iopsstor/scratch/cscs/tong/share/xianrong_liu/build-nanogpt

srun --uenv=pytorch/v2.8.0:v1 --view=default bash -c " \
    source /iopsstor/scratch/cscs/tong/share/xianrong_liu/.venv/bin/activate && \
    source /iopsstor/scratch/cscs/tong/share/xianrong_liu/build-nanogpt/cache_env_setup.sh && \
    python -m torch.distributed.run --standalone --nproc_per_node=4 train_gpt2_no_warmup.py \
"
