#!/bin/bash
#SBATCH -p big_compute
#SBATCH -N 1
#SBATCH -c 128
#SBATCH -t 06:00:00
#SBATCH --job-name="eval_td3"
#SBATCH -o eval_td3.%j.out
#SBATCH -e eval_td3.%j.err

cd ~/research/DDPG_MQTT_125/src
./run_evaluation_and_collect.sh td3 374946 20
