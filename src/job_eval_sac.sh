#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH --job-name="eval_sac"
#SBATCH -o eval_sac.%j.out
#SBATCH -e eval_sac.%j.err

cd ~/research/DDPG_MQTT_125/src
./run_evaluation_and_collect.sh sac 374945
