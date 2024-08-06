#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=23:00:00     # DD-HH:MM:SS
#SBATCH --mail-user=leokx.cheng@mail.utoronto.ca
#SBATCH --mail-type=END
#SBATCH --array=1-3

module load python/3.10
source $PROJECT/equivariant_exp/venv/bin/activate
export MPLCONFIGDIR=/scratch/a/aspuru/chengl43/mplconfigdir

param_store=`pwd`/args.txt
blocks=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
layers=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $2}')
img_dim=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $3}')

echo "blocks: $blocks"
echo "layers: $layers"
echo "img_dim: $img_dim"

srun python -u mlp2gcnn.py --blocks $blocks --layers $layers --bs 1000 --dataset 0 --img_dim $img_dim
