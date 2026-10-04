#!/bin/bash
#SBATCH -p big_compute
#SBATCH -N 1
#SBATCH -c 128
#SBATCH -t 06:00:00
#SBATCH --job-name="eval_sac"
#SBATCH -o eval_sac.%j.out
#SBATCH -e eval_sac.%j.err

cd ~/research/DDPG_MQTT_125/src
./run_evaluation_and_collect.sh sac 374945 20
