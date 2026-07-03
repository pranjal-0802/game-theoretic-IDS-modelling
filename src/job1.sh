#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -c 20
#SBATCH -t 4-00:00 
#SBATCH --job-name="d_125"
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err

python3 evaluate_ddpg.py snort rl 1000 125 rl 125 20
