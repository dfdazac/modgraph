#!/bin/bash
#SBATCH --job-name=link_pred
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

PROJ_FOLDER=modgraph
LOG_FOLDER=logs

# Copy data to scratch
cp -r $HOME/$PROJ_FOLDER $TMPDIR
cd $TMPDIR/$PROJ_FOLDER

# Run experiment
source activate pygeom
srun python -u train.py modular_search_link_pred with \
dataset_str="citeseer" \
lr=0.001 \
n_exper=1

cp -r $TMPDIR/$PROJ_FOLDER/$LOG_FOLDER $HOME/$PROJ_FOLDER
