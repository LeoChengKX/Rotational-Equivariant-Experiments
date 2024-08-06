#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=23:00:00     # DD-HH:MM:SS
#SBATCH --mail-user=leokx.cheng@mail.utoronto.ca
#SBATCH --mail-type=END
#SBATCH --array=1

module load python/3.10
source $PROJECT/rot_equiv_exp/venv/bin/activate
export MPLCONFIGDIR=/scratch/a/aspuru/chengl43/mplconfigdir

srun python -u mlp2gcnn_P4P4.py --blocks 9 --layers 1 --lr 0.005 --bs 1000 --dataset 0
