#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -c 20
#SBATCH -t 4-00:00
#SBATCH --job-name="do_td3"
#SBATCH -o do_td3.%j.out
#SBATCH -e do_td3.%j.err

cd ~/research/DDPG_MQTT_125/src
~/.conda/envs/Alerts/bin/python -u double_oracle.py snort 1000 125 20 td3
