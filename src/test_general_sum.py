from model import Model, AlertType, AttackType, PoissonDistribution
from test import test_attack_action, test_defense_newest

print("=== Test 1: General-sum (gain != loss) ===")
alert_types = [AlertType(1.0, PoissonDistribution(10), "t1")]
attack_types = [
    AttackType([5.0], 1, [0.8], "a1", gain=[3.0]),
    AttackType([2.0], 1, [0.2], "a2", gain=[8.0]),
]
model = Model(1, alert_types, attack_types, 10.0, 5.0)
state = Model.State(model)

for i in range(5):
    state = model.next_state('old', state, test_defense_newest, test_attack_action)
    print(f"Step {i+1}: U={state.U:.3f}  U_defender={state.U_defender:.3f}  U_attacker={state.U_attacker:.3f}")
    print(f"         U == U_defender: {abs(state.U - state.U_defender) < 0.001}")

print("\n=== Test 2: Zero-sum default (gain=loss) ===")
attack_types_zs = [
    AttackType([5.0], 1, [0.8], "a1"),
    AttackType([2.0], 1, [0.2], "a2"),
]
model_zs = Model(1, alert_types, attack_types_zs, 10.0, 5.0)
state_zs = Model.State(model_zs)

for i in range(5):
    state_zs = model_zs.next_state('old', state_zs, test_defense_newest, test_attack_action)
    total = state_zs.U_defender + state_zs.U_attacker
    print(f"Step {i+1}: U_defender={state_zs.U_defender:.3f}  U_attacker={state_zs.U_attacker:.3f}  sum={total:.3f}")

print("\nDone. Sum should be near 0 in Test 2.")
