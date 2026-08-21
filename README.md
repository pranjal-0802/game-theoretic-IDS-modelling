# Game-Theoretic IDS Modelling

A research codebase for computing **defender–attacker equilibria in the alert-prioritization problem** faced by network intrusion detection systems (IDS). It combines a **double-oracle algorithm** with **deep reinforcement learning (DDPG / SAC / TD3) best responses** to approximate Nash equilibria in games that can be either **zero-sum** or **general-sum**, and evaluates the resulting defense policies against a range of baselines (uniform, Suricata/Snort-style priority rules, greedy attackers, etc.) on datasets derived from real IDS alert logs.

This work extends prior zero-sum alert-prioritization models (Suricata / Snort / credit-fraud datasets) to a **general-sum formulation**, where the attacker's gain is decoupled from the defender's loss, and studies how equilibrium selection (e.g. defender-optimal vs. welfare-maximizing) affects outcomes.

## Background

An IDS raises large volumes of alerts, only a fraction of which correspond to real attacks; investigating every alert is infeasible under a fixed budget. This is modelled as a repeated game:

- **Defender**: chooses how many alerts of each type/age to investigate (`δ`) under a budget `B`.
- **Attacker**: chooses which attack types to mount, at what probability (`α`), under a budget `D`, to evade detection and inflict loss / gain reward.

The `Model` class (`src/model.py`) encodes this as a stochastic game with:
- alert types (`AlertType`: investigation cost + false-alert distribution),
- attack types (`AttackType`: loss to defender, cost to mount, per-type alert-triggering probabilities, and optionally a separate `gain` to the attacker for general-sum games),
- a state transition (`Model.next_state`) that resolves investigations, attacks, and new alerts each time step.

Solving the game exactly is intractable over the full strategy space, so the codebase uses a **double-oracle** loop: iteratively compute a mixed Nash equilibrium over a restricted set of pure strategies, then train new best-response policies (attacker and defender) against the current equilibrium via deep RL, and add them to the strategy pool until convergence.

## Repository Structure

```
.
├── src/
│   ├── model.py               # Core game model: alert/attack types, state, transition dynamics
│   ├── model_test.py          # Variant of model.py used by some evaluation scripts
│   ├── test.py                 # Test model instances (Snort/Suricata/fraud datasets) and baseline policies
│   ├── test_general_sum.py    # Test instances for the general-sum formulation
│   ├── config.py / project.conf  # Central hyperparameter / experiment configuration
│   ├── double_oracle.py       # Double-oracle equilibrium computation (uses nashpy for NE support enumeration)
│   ├── ddpg.py / ddpg_fn.py   # DDPG-based best-response oracles (attacker & defender)
│   ├── sac.py / sac_fn.py     # Soft Actor-Critic best-response oracles
│   ├── td3_v3_1.py            # TD3 best-response oracle
│   ├── evaluate*.py           # Evaluation harnesses that pit defense vs. attack policies
│   │                          #   (evaluate.py, evaluate_ddpg.py, evaluate_sac.py, evaluate_td3.py,
│   │                          #    evalddpg.py, evalsac.py, evaltd3.py)
│   ├── listutils.py           # List flattening/normalization helpers, Ornstein–Uhlenbeck noise
│   └── job1.sh                # Example SLURM batch script for running an evaluation on a cluster
└── model/                     # Output directory: trained policies, payoff matrices, equilibrium strategies
    ├── attack_models/
    ├── converge/
    ├── attacker-<exp>-<iter>-<trial>/
    └── defender-<exp>-<iter>-<trial>/
```

## The Game Model (`src/model.py`)

- `Model(horizon, alert_types, attack_types, def_budget, adv_budget)` — defines a stochastic game instance.
- `Model.State` — tracks, per time step: uninvestigated alerts (`N`), undetected attacks (`M`), true-alert indicators (`R`), and cumulative defender/attacker utility (`U`, `U_defender`, `U_attacker`).
- `next_state(mode, state, delta, alpha)` — advances the game by one step:
  1. resolves which ongoing attacks get investigated (and hence caught),
  2. accrues defender loss / attacker gain (mode `'old'`: full loss/gain each step an attack is live; mode `'new'`: only the loss delta when an attack is caught or ages out),
  3. samples new attacks according to policy `alpha`,
  4. samples new alerts (true + false-positive) for the next step.
- `is_feasible_investigation` / `make_investigation_feasible` and `is_feasible_attack` / `make_attack_feasible` enforce and project actions onto the defender/attacker budget constraints.

Pre-built test instances in `src/test.py` include `test_model_snort`, `test_model_suricata`, and `test_model_fraud`, calibrated to alert-volume statistics from real Snort/Suricata deployments and a credit-card fraud dataset. Baseline policies are also provided for both sides, e.g. `test_defense_newest`, `test_defense_proportion`, `test_defense_suricata`/`test_defense_snort` (priority-order heuristics), `test_defense_aics`/`test_defense_icde` (prior published baselines), and `test_attack_action` (budget-uniform attacker).

## Solving for Equilibria (`src/double_oracle.py`)

`double_oracle(model, exper_index)` implements the main loop:

1. Compute a mixed-strategy Nash equilibrium of the current (small) payoff-matrix game via `find_mixed_NE`, which wraps `nashpy`'s support enumeration.
   - `selection='defender_max'` (default, kept for backward compatibility) picks the equilibrium maximizing defender utility — note this is a biased tie-break when multiple equilibria exist.
   - `selection='welfare_max'` picks the equilibrium maximizing total (defender + attacker) welfare, a more neutral alternative.
   - `report_range=True` logs the min/max/mean defender utility across *all* computed equilibria, to expose how much the choice of equilibrium matters.
2. Train `N_TRIAL` new DDPG-based attacker and defender best-response policies (`AttackerOracle` / `DefenderOracle` from `ddpg.py`) against the opponent's current mixed strategy, keeping the best of each.
3. Expand the payoff matrices with the new pure strategies (`update_profile`) and repeat until neither side's new best response improves on the current equilibrium utility, or `MAX_ITERATION` is reached.
4. Persist the final mixed strategies and per-trial utilities to `../model/` as pickle files.

Run as a script:

```bash
cd src
python3 double_oracle.py <model_name> <def_budget> <adv_budget> <n_experiment>
# e.g.
python3 double_oracle.py snort 1000 125 5
```

`model_name` is one of `suricata`, `fraud`, `snort`. Multiple experiments (`n_experiment`) are run in parallel across CPU cores via `multiprocessing`.

## Best-Response Oracles

Three RL algorithms are available to train best responses for either player, all implemented against the TensorFlow 1.x API (`tf.Session`, `tf.placeholder`):

| Algorithm | File(s) |
|---|---|
| DDPG | `ddpg.py`, `ddpg_fn.py` |
| SAC | `sac.py`, `sac_fn.py` |
| TD3 | `td3_v3_1.py` |

Hyperparameters (learning rates, discount `gamma`, soft-update `tau`, hidden layer sizes, replay memory size/init, batch size, exploration schedule, etc.) are centralized in `src/project.conf` and loaded via `src/config.py`.

## Evaluation

Once policies are trained (or using the built-in baselines), the `evaluate*.py` scripts pit a chosen defense policy against a chosen attack policy and report average discounted utility:

```bash
python3 evaluate_ddpg.py <model_name> <defense> <def_budget> <attack> <adv_budget> <n_experiment>
```

Supported `defense` values include `uniform`, `rio`, `proportion`, `gain`, `suricata`/`snort`, or a trained `rl` policy loaded from a saved profile. Supported `attack` values include `uniform`, `greedy`, or trained `rl`/`sac`/`ddpg` policies. Variants (`evaluate.py`, `evaluate_sac.py`, `evaluate_td3.py`, `evalddpg.py`, `evalsac.py`, `evaltd3.py`) swap in the corresponding oracle/algorithm.

`src/job1.sh` shows an example SLURM submission for running an evaluation on an HPC cluster.

## Getting Started

**Requirements**
- Python 3.6 (see `.python-version`; the TensorFlow 1.x API used throughout is not compatible with modern TF)
- `tensorflow` (1.x), `numpy`, `scipy`, `nashpy`

```bash
pip install "tensorflow<2" numpy scipy nashpy
```

**Quick sanity check of the game dynamics:**

```bash
cd src
python3 test.py
```

**Run a full double-oracle experiment:**

```bash
cd src
python3 double_oracle.py snort 1000 125 1
```

Outputs (trained policy checkpoints, payoff matrices, and pickled equilibrium strategies/utilities) are written under `../model/`.

## Status

This is an active research codebase (see `project.conf`'s `[game] game_type = general_sum`), currently being extended from a zero-sum formulation to general-sum games — including work on principled equilibrium selection (`welfare_max` vs. `defender_max`) when multiple Nash equilibria exist. Expect rough edges: some scripts assume specific datasets/paths, and legacy TF1 APIs require pinned dependency versions.

## Acknowledgements

Core game model and double-oracle framework originally authored by Aron Laszka and Liang Tong; extended with a general-sum formulation and equilibrium-selection analysis as part of ongoing research.
