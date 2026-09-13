#!/usr/bin/env python3
"""
Quick sanity check for gain/cost parameter choices in the general-sum snort model.
Run this BEFORE a full double_oracle.py training run to check whether different
attack strategies produce meaningfully different U_attacker outcomes.

If even hand-picked "obviously good" vs "obviously bad" attacker policies produce
near-identical U_attacker, RL has nothing to learn and will converge trivially
(exactly what happened with gain=[2.0], gain=[3.5]).

Usage: place in src/, then:
    python test_gain_sensitivity.py
"""

from model import Model, AlertType, AttackType, PoissonDistribution
from test import test_defense_newest
import numpy as np

N_STEPS = 20
N_TRIALS = 30  # average over multiple runs since next_state() has randomness


def run_policy(model, attack_alpha, n_steps=N_STEPS, n_trials=N_TRIALS):
    """Run a fixed attack probability vector against test_defense_newest, return average U_attacker and U_defender."""
    def fixed_attack_policy(model, state):
        return attack_alpha

    atk_totals = []
    def_totals = []
    for _ in range(n_trials):
        state = Model.State(model)
        for _ in range(n_steps):
            state = model.next_state('old', state, test_defense_newest, fixed_attack_policy)
        atk_totals.append(state.U_attacker)
        def_totals.append(state.U_defender)
    return np.mean(atk_totals), np.mean(def_totals)


def sweep(def_budget, adv_budget, gain_values, label):
    print(f"\n{'='*70}")
    print(f"CONFIG: {label}  (def_budget={def_budget}, adv_budget={adv_budget}, gain={gain_values})")
    print('='*70)

    alert_types = [AlertType(1.0, PoissonDistribution(10274000), "t1"),
                   AlertType(1.0, PoissonDistribution(0), "t2")]
    attack_types = [
        AttackType([3.6], 135.0, [862, 0], "a1", gain=[gain_values[0]]),
        AttackType([1.4], 25.0, [18785, 281], "a2", gain=[gain_values[1]]),
    ]
    model = Model(1, alert_types, attack_types, def_budget, adv_budget)

    # Candidate attacker policies to compare -- these are NOT trained, just fixed
    # hand-picked strategies to see if the utility landscape has any texture at all.
    policies = {
        "all_attack_a1":    [1.0, 0.0],
        "all_attack_a2":    [0.0, 1.0],
        "no_attack":        [0.0, 0.0],
        "uniform_50_50":    [0.5, 0.5],
        "aggressive_both":  [1.0, 1.0],
    }

    results = {}
    for name, alpha in policies.items():
        atk_u, def_u = run_policy(model, alpha)
        results[name] = (atk_u, def_u)
        print(f"  {name:20s} atk_alpha={alpha}  ->  U_attacker={atk_u:9.3f}   U_defender={def_u:9.3f}")

    atk_values = [v[0] for v in results.values()]
    spread = max(atk_values) - min(atk_values)
    print(f"\n  Spread across policies (U_attacker max-min): {spread:.3f}")
    if spread < 5.0:
        print("  --> FLAT landscape: policies barely differ. RL will likely converge trivially.")
    else:
        print("  --> Policies produce meaningfully different outcomes. Worth training RL on this.")
    return results


if __name__ == "__main__":
    # 1. Current values (what actually ran and converged trivially)
    sweep(1000, 125, [2.0, 3.5], "CURRENT (illustrative, converged flat)")

    # 2. Rescaled: gain now clearly exceeds cost for a successful attack
    sweep(1000, 125, [200.0, 40.0], "RESCALED: gain > cost")

    # 3. Asymmetric: a1 clearly profitable, a2 clearly not
    sweep(1000, 125, [250.0, 5.0], "ASYMMETRIC: a1 profitable, a2 a trap")

    print("\nDone. Compare 'Spread' values across configs above to pick better gain parameters.")