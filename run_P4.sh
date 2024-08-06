#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=23:00:00     # DD-HH:MM:SS
#SBATCH --mail-user=leokx.cheng@mail.utoronto.ca
#SBATCH --mail-type=END
#SBATCH --array=1-2

module load python/3.10
source $PROJECT/rot_equiv_exp/venv/bin/activate
export MPLCONFIGDIR=/scratch/a/aspuru/chengl43/mplconfigdir

param_store=`pwd`/args_P4.txt
blocks=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
layers=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $2}')
filter=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $3}')
img_dim=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $4}')
emb_dim=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $5}')


echo "blocks: $blocks"
echo "layers: $layers"
echo "filter: $filter"
echo "emb_dim: $emb_dim"

srun python -u mlp2gcnn_P4.py --blocks $blocks --layers $layers --provide_filter $filter --emb_dim $emb_dim --img_dim $img_dim --lr 0.005 --bs 500 --dataset 0 
