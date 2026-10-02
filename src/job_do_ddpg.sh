#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -c 20
#SBATCH -t 4-00:00
#SBATCH --job-name="do_ddpg"
#SBATCH -o do_ddpg.%j.out
#SBATCH -e do_ddpg.%j.err

cd ~/research/DDPG_MQTT_125/src
~/.conda/envs/Alerts/bin/python -u double_oracle.py snort 1000 125 20 ddpg
