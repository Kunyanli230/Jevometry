# Run inside a copy of the upstream repo with a torch venv: python retrain_upstream.py 150
"""Reduced IMAS² retrain: reported selection order 4,2,8,6,3; N iters each."""
import json, sys, time, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.show = lambda: None
import model_training as mt
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
order = [4, 2, 8, 6, 3]
mt.set_random_seeds(1337)
fixed, paths, timing = [], [], {}
for ph, sn in enumerate(order, 1):
    net = mt.PolicyNetwork(len(mt.prod_pomdp.observations), mt.prod_pomdp.action_size, 64, 15)
    tr = mt.MultiAgentPolicyGradient("retrain", f"s{sn}_{ph}", 15, 100, N, 0.001, sn, net, list(fixed), list(paths))
    t0 = time.time(); tr.train(); timing[sn] = time.time() - t0
    print("PHASE", ph, sn, timing[sn], flush=True)
    fixed.append(sn); paths.append(f"./data_retrain/Values/policy_net_s{sn}_{ph}.pkl")
# rollouts: last-trained sensor primary, others fixed (all trained)
mt.set_random_seeds(0)
net = mt.PolicyNetwork(len(mt.prod_pomdp.observations), mt.prod_pomdp.action_size, 64, 15)
import torch; net.load_state_dict(torch.load(paths[-1])); net.eval()
tr = mt.MultiAgentPolicyGradient("retrain", "eval", 15, 100, 1, 0.001, order[-1], net, fixed[:-1], paths[:-1])
t0 = time.time(); st, obs, act = tr.sample_data_network(500, 15); roll_t = time.time() - t0
ent = [pickle.load(open(f"data_retrain/Values/entropy_s{sn}_{ph}", "rb")) for ph, sn in enumerate(order, 1)]
out = {"order_primary_first": [order[-1]] + fixed[:-1], "iters": N, "M": 100, "T": 15,
       "training_time_s": timing, "rollout_time_s": roll_t,
       "final_entropy_last50": [sum(e[-50:]) / len(e[-50:]) for e in ent],
       "trajectories": [{"states": [[list(s[0][0]), s[0][1]] if s != "sink1" else None for s in sl],
                         "acts": [list(a) for a in al]} for sl, al in zip(st, act)]}
json.dump(out, open("retrain_rollouts.json", "w"))
print("DONE", flush=True)
