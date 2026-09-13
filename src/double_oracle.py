#!/usr/bin/env python3
# Author: Liang Tong

from scipy import optimize as op
from listutils import *
from model import Model
from test import *
from listutils import *
from sac import DefenderOracle
from sac import AttackerOracle
from config import config

import multiprocessing
import numpy as np
import random
import logging
import pickle
import sys
import csv
import tensorflow as tf
import os
import glob
import shutil
import pickle
import nashpy as nash 

os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

"""Implementation of double-oracle algorithms."""

##################### CONFIGURATION ######################
MAX_EPISODES = config.getint('parameter', 'max_episodes')
MAX_STEPS = config.getint('parameter', 'max_ep_steps')
GAMMA = config.getfloat('parameter', 'gamma')
MAX_ITERATION = config.getint('double-oracle', 'max_iteration')
N_TRIAL = config.getint('double-oracle', 'n_trial')
##########################################################

def find_mixed_NE(payoff_def, payoff_atk, selection='defender_max', report_range=False):
    """
    Compute a mixed-strategy Nash equilibrium for the general-sum restricted game.
    :param payoff_def: 2D array. Row = defender, column = attacker. Defender's payoffs.
    :param payoff_atk: 2D array, same shape. Attacker's payoffs.
    :param selection: Which equilibrium to return when multiple exist.
        'defender_max' (default, original behaviour): pick the equilibrium that
            maximizes the defender's utility. NOTE: this is a biased tie-break,
            not a neutral one -- it systematically favors the defender whenever
            several equilibria exist. Kept as default only for backward
            compatibility with prior runs.
        'welfare_max': pick the equilibrium maximizing the sum of both players'
            utilities (a more neutral, Pareto-motivated tie-break).
    :param report_range: If True, also logs (via `logging.info`) the min/max/mean
        defender utility across ALL computed equilibria, so the spread introduced
        by equilibrium selection is visible in the log regardless of which
        equilibrium is ultimately returned.
    :return: attack_strategy (list), defense_strategy (list), defense_utility (float)
    """
    game = nash.Game(payoff_def, payoff_atk)
    equilibria = list(game.support_enumeration())

    def def_utility_of(eq):
        sigma_def, sigma_atk = eq
        return np.dot(np.dot(sigma_def, payoff_def), sigma_atk)

    def atk_utility_of(eq):
        sigma_def, sigma_atk = eq
        return np.dot(np.dot(sigma_def, payoff_atk), sigma_atk)

    def welfare_of(eq):
        return def_utility_of(eq) + atk_utility_of(eq)

    if len(equilibria) == 0:
        n_def = payoff_def.shape[0]
        n_atk = payoff_def.shape[1]
        defense_strategy = [1.0 / n_def] * n_def
        attack_strategy = [1.0 / n_atk] * n_atk
        defense_utility = np.dot(np.dot(defense_strategy, payoff_def), attack_strategy)
        return attack_strategy, defense_strategy, defense_utility

    if report_range:
        def_utils = [def_utility_of(eq) for eq in equilibria]
        logging.info(
            "find_mixed_NE: {} equilibria found. Defender utility range: "
            "min={:.4f}, max={:.4f}, mean={:.4f}".format(
                len(equilibria), min(def_utils), max(def_utils),
                sum(def_utils) / len(def_utils)))

    if selection == 'welfare_max':
        best_eq = max(equilibria, key=welfare_of)
    elif selection == 'defender_max':
        best_eq = max(equilibria, key=def_utility_of)
    else:
        raise ValueError("Unknown selection rule: {}".format(selection))

    sigma_def, sigma_atk = best_eq

    defense_strategy = list(sigma_def)
    attack_strategy = list(sigma_atk)
    defense_utility = def_utility_of(best_eq)

    return attack_strategy, defense_strategy, defense_utility

def get_payoff_mixed(model, attack_profile, defense_profile, attack_strategy, defense_strategy):
    """
    :return: (defender_payoff, attacker_payoff) — expected discounted rewards for both players.
    """
    total_discount_reward_def = 0.0
    total_discount_reward_atk = 0.0

    attack_policies = np.random.choice(attack_profile, MAX_EPISODES, p=attack_strategy)
    defense_policies = np.random.choice(defense_profile, MAX_EPISODES, p=defense_strategy)

    initial_state = Model.State(model)

    csv_path = os.path.join(os.getcwd(), 'episode_rewards.csv')
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Episode', 'Defender Reward', 'Attacker Reward', 'Sum'])
        for i in range(MAX_EPISODES):
            state = initial_state
            episode_reward_def = 0.0
            episode_reward_atk = 0.0
            defense_policy = defense_policies[i]
            attack_policy = attack_policies[i]
            for j in range(MAX_STEPS):
                next_state = model.next_state('old', state, defense_policy, attack_policy)
                def_loss = next_state.U_defender - state.U_defender
                atk_gain = next_state.U_attacker - state.U_attacker
                state = next_state
                episode_reward_def += GAMMA**j * (-1.0 * def_loss)
                episode_reward_atk += GAMMA**j * atk_gain
            total_discount_reward_def += episode_reward_def
            total_discount_reward_atk += episode_reward_atk
            writer.writerow([
                i + 1,
                float(episode_reward_def),
                float(episode_reward_atk),
                float(episode_reward_def + episode_reward_atk),
            ])

    ave_discount_reward_def = total_discount_reward_def / MAX_EPISODES
    ave_discount_reward_atk = total_discount_reward_atk / MAX_EPISODES
    return ave_discount_reward_def, ave_discount_reward_atk

def get_payoff(model, attack_policy, defense_policy):
    """
    :return: (defender_payoff, attacker_payoff)
    """
    ave_discount_reward_def, ave_discount_reward_atk = get_payoff_mixed(
        model, [attack_policy], [defense_policy], [1.0], [1.0])
    return ave_discount_reward_def, ave_discount_reward_atk

def update_profile(model, payoff_def, payoff_atk, attack_profile,
        defense_profile, attack_policy, defense_policy):
    """
    Function for updating both payoff matrices and the action profile of defender and attacker.
    :param model: Model of the alert prioritization problem (i.e., Model object).
    :param payoff_def: Two dimensional array. Defender's payoff matrix. Row = defender, column = attacker.
    :param payoff_atk: Two dimensional array. Attacker's payoff matrix. Same shape as payoff_def.
    :param attack_profile: List of attack policies.
    :param defense_profile: List of defense policies.
    :param attack_policy: New pure strategy of the attacker
    :param defense_policy: New pure strategy of the defender
    :return: updated payoff_def, payoff_atk, attack_profile, defense_profile
    """
    # A new row and column will be added to both payoff matrices
    new_payoff_col_def = np.array([])
    new_payoff_col_atk = np.array([])
    new_payoff_row_def = np.array([])
    new_payoff_row_atk = np.array([])

    # First get the new column (new attacker vs all existing defenders)
    for i in range(len(defense_profile)):
        pd, pa = get_payoff(model, attack_policy, defense_profile[i])
        new_payoff_col_def = np.append(new_payoff_col_def, pd)
        new_payoff_col_atk = np.append(new_payoff_col_atk, pa)
    new_payoff_col_def = np.expand_dims(new_payoff_col_def, axis=0)
    new_payoff_col_atk = np.expand_dims(new_payoff_col_atk, axis=0)
    attack_profile.append(attack_policy)
    payoff_def = np.concatenate((payoff_def, new_payoff_col_def.T), axis=1)
    payoff_atk = np.concatenate((payoff_atk, new_payoff_col_atk.T), axis=1)

    # Second, get the new row (all attackers, including new one, vs new defender)
    for j in range(len(attack_profile)):
        pd, pa = get_payoff(model, attack_profile[j], defense_policy)
        new_payoff_row_def = np.append(new_payoff_row_def, pd)
        new_payoff_row_atk = np.append(new_payoff_row_atk, pa)
    new_payoff_row_def = np.expand_dims(new_payoff_row_def, axis=0)
    new_payoff_row_atk = np.expand_dims(new_payoff_row_atk, axis=0)
    defense_profile.append(defense_policy)
    payoff_def = np.concatenate((payoff_def, new_payoff_row_def), axis=0)
    payoff_atk = np.concatenate((payoff_atk, new_payoff_row_atk), axis=0)

    return payoff_def, payoff_atk, attack_profile, defense_profile

def promote_checkpoint(checkpoint_root, role, exper_index, iteration_index, trial_index):
    source_dir = os.path.join(
        checkpoint_root,
        '{}-{}-{}-{}'.format(role, exper_index, iteration_index, trial_index))
    target_dir = os.path.join(
        checkpoint_root,
        '{}-{}-{}'.format(role, exper_index, iteration_index))
    os.makedirs(target_dir, exist_ok=True)
    for source_path in glob.glob(os.path.join(source_dir, 'ddpg.ckpt.*')):
        target_path = os.path.join(target_dir, os.path.basename(source_path))
        shutil.copy2(source_path, target_path)

def double_oracle(model, exper_index, model_name, def_budget, adv_budget):
    checkpoint_root = os.path.join(
        '../model/converge',
        '{}_{}_{}_do'.format(model_name, int(def_budget), int(adv_budget)))
    os.makedirs(checkpoint_root, exist_ok=True)
    attack_profile = []
    defense_profile = []
    payoff_record = []

    initial_payoff_def, initial_payoff_atk = get_payoff(model, test_attack_action, test_defense_newest)
    payoff_def = np.array([[initial_payoff_def]])
    payoff_atk = np.array([[initial_payoff_atk]])

    attack_profile = [test_attack_action]
    defense_profile = [test_defense_newest]
    initial_defense_size = payoff_def.shape[0]
    initial_attack_size = payoff_def.shape[1]

    for i in range(MAX_ITERATION):
        attack_strategy, defense_strategy, utility = find_mixed_NE(
            payoff_def, payoff_atk, selection='defender_max', report_range=True)
        payoff_record.append(utility)

        # Current equilibrium utility for BOTH players, computed on matrices as they
        # stand this iteration (before expansion) — this is what best responses are
        # compared against.
        current_defense_utility = utility
        current_attack_utility = np.dot(np.dot(defense_strategy, payoff_atk), attack_strategy)

        print("###########################################################################################")
        print("Iteration", i)
        print("Current defender payoff matrix:")
        print(payoff_def)
        print("Current attacker payoff matrix:")
        print(payoff_atk)
        print("Attacker's mixed strategy:", attack_strategy)
        print("Defender's mixed strategy:", defense_strategy)

        logging.info("Iteration {}: def={} att={}".format(i, current_defense_utility, current_attack_utility))

        attack_response = []
        attack_utility = []
        defense_response = []
        defense_utility = []
        for k in range(N_TRIAL):
            attack_response.append(AttackerOracle(model, defense_profile, defense_strategy, exper_index, i, k, checkpoint_root))
            defense_response.append(DefenderOracle(model, attack_profile, attack_strategy, exper_index, i, k, checkpoint_root))
            attack_utility.append(attack_response[k].agent.utility)
            defense_utility.append(defense_response[k].agent.utility)

        pickle.dump(defense_utility, open(os.path.join(checkpoint_root, "defender-utility-{}.pickle".format(exper_index)), 'wb'))

        attack_index = attack_utility.index(max(attack_utility))
        defense_index = defense_utility.index(max(defense_utility))
        attack_policy = attack_response[attack_index].agent.policy
        defense_policy = defense_response[defense_index].agent.policy

        promote_checkpoint(checkpoint_root, 'attacker', exper_index, i, attack_index)
        promote_checkpoint(checkpoint_root, 'defender', exper_index, i, defense_index)

        for k in range(N_TRIAL):
            if k != defense_index:
                shutil.rmtree(os.path.join(checkpoint_root, 'defender-{}-{}-{}'.format(exper_index, i, k)), ignore_errors=True)
            if k != attack_index:
                shutil.rmtree(os.path.join(checkpoint_root, 'attacker-{}-{}-{}'.format(exper_index, i, k)), ignore_errors=True)

        payoff_def, payoff_atk, attack_profile, defense_profile = update_profile(
            model, payoff_def, payoff_atk, attack_profile, defense_profile,
            attack_policy, defense_policy)

        # New defender policy's payoff against attacker's mixed strategy (new row, excluding last col which is the new attacker col)
        defense_pure_utility = np.dot(payoff_def[i + initial_defense_size][:-1], attack_strategy)
        # New attacker policy's payoff against defender's mixed strategy (new column, excluding last row which is the new defender row)
        attack_pure_utility = np.dot(payoff_atk[:, i + initial_attack_size][:-1], defense_strategy)

        if defense_pure_utility <= current_defense_utility and attack_pure_utility <= current_attack_utility:
            pickle.dump(defense_strategy, open(os.path.join(checkpoint_root, "defender-strategy-{}.pickle".format(exper_index)), 'wb'))
            pickle.dump(attack_strategy, open(os.path.join(checkpoint_root, "attacker-strategy-{}.pickle".format(exper_index)), 'wb'))
            break
        if i == MAX_ITERATION - 1:
            pickle.dump(defense_strategy, open(os.path.join(checkpoint_root, "defender-strategy-{}.pickle".format(exper_index)), 'wb'))
            pickle.dump(attack_strategy, open(os.path.join(checkpoint_root, "attacker-strategy-{}.pickle".format(exper_index)), 'wb'))

    return payoff_record[-1]

def test_mixed_NE():
    """
    Test find_mixed_NE with Rock-Paper-Scissors (zero-sum): mixed NE should be ~0.33 each.
    """
    payoff = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
    attack_strategy, defense_strategy, utility = find_mixed_NE(payoff, -payoff)
    print(attack_strategy, defense_strategy)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s / %(levelname)s: %(message)s', level=logging.DEBUG)
    logging.info("Experiment starts.")
    if len(sys.argv) < 5:
        print("python do_h1_mul.py [model_name] [def_budget] [adv_budget] [n_experiment]")
        sys.exit(1)

    model_name = sys.argv[1]
    def_budget = float(sys.argv[2])
    adv_budget = float(sys.argv[3])
    n_experiment = int(sys.argv[4])

    if model_name == 'suricata':
        model = test_model_suricata(def_budget, adv_budget)
    elif model_name == 'fraud':
        model = test_model_fraud(def_budget, adv_budget)
    elif model_name == 'snort':
        model = test_model_snort(def_budget, adv_budget)

    def evaluation(exper_index):
        random_seed = exper_index
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        do_utility = double_oracle(model, exper_index, model_name, def_budget, adv_budget)
        return do_utility

    cores = multiprocessing.cpu_count()
    #cores = 1	
    pool = multiprocessing.Pool(processes=cores)
    utilities = []
    for do_utility in pool.imap(evaluation, range(n_experiment)):
        utilities.append(do_utility)    
    logging.info("The utility of the agent:")
    print(utilities)

