#!/bin/bash
#SBATCH --job-name=node_class
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

PROJ_FOLDER=modgraph

# Copy data to scratch
cp -r $HOME/$PROJ_FOLDER $TMPDIR
cd $TMPDIR/$PROJ_FOLDER

# Run experiment
source activate pygeom
srun python -u train.py node_class_experiments with \
dataset_str="coauthorcs" \
encoder_str="mlp" \
repr_str="euclidean_inner" \
loss_str="hinge_loss" \
sampling_str="first_neighbors" \
lr=0.001 \
n_exper=20
