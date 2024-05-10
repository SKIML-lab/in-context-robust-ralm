#!/bin/bash
#SBATCH -J load
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH -o log/load.out
#SBATCH -e log/load.err
#SBATCH --time 48:00:00

export PYTHONPATH=../
python3 load.py