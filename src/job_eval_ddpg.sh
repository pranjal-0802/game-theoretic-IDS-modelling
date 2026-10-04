#!/bin/bash
#SBATCH -p big_compute
#SBATCH -N 1
#SBATCH -c 128
#SBATCH -t 06:00:00
#SBATCH --job-name="eval_ddpg"
#SBATCH -o eval_ddpg.%j.out
#SBATCH -e eval_ddpg.%j.err

cd ~/research/DDPG_MQTT_125/src
./run_evaluation_and_collect.sh ddpg 374947 20
