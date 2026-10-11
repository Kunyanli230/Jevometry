"""IMAS² grid: tr(ΔI) vs fused posterior entropy / accuracy across seven rules,
plus Fisher-vs-MI scoring of all C(9,5)=126 placements.

Writes papers/coord-fisher/data/imas2_*.{csv,json}.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import system as S  # noqa: E402
from jevometry.systems.pushforward import information_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "papers" / "coord-fisher" / "data"
SEED = 0
N_TRAJ = 500
N_RANDOM_ACTS = 8
RETRAIN_JSON = Path("/tmp/imas2-run/retrain_rollouts.json")
MAIN_FIELDS = ["rule", "policy_type", "dynamics", "sensor_range", "tr_delta_I", "cond_entropy",
               "accuracy", "seed", "n_trajectories", "selection_time_s", "training_time_s", "tr_I_Z", "status"]


# ---------------- orientation policies ----------------
def fixed_orientation(model: S.GridModel) -> tuple[str, ...]:
    mu = model.occupancy(model.theta0).sum(axis=(0, 1))
    return tuple(max(S.ACTIONS, key=lambda a: float(mu[cov[a]].sum())) for cov in model.coverage)


def action_schedule(model, policy_type, rng, n_traj, retrain=None):
    """acts[m][t] tuple per trajectory/step."""
    T = S.HORIZON
    if policy_type == "fixed_orientation":
        a = fixed_orientation(model)
        return [[a] * T for _ in range(n_traj)]
    if policy_type == "random_orientation":
        return [[tuple(rng.choice(S.ACTIONS, size=len(model.sensors))) for _ in range(T)] for _ in range(n_traj)]
    return retrain["acts"]


# ---------------- Fisher ----------------
def fisher_by_rule(model, steps_acts, maps):
    """Σ_t E_acts[I_Z - I_A]; steps_acts[t] = list of (acts, weight)."""
    k = len(model.sensors)
    tot = {n: np.zeros((2, 2)) for n, _ in maps}
    iz = np.zeros((2, 2))
    ok = True
    for t, pairs in enumerate(steps_acts):
        for acts, w in pairs:
            p, j = S.step_joint_and_jacobian(model, t, acts)
            jd, jj = S.joint_distribution(p, j, k)
            for n, amap in maps:
                r = information_loss(jd, jj, amap, parameter_names=S.PARAMETERS)
                if r.difference is None:
                    ok = False
                    continue
                ok &= r.status.value == "ok"
                tot[n] += w * r.difference
                if n == "near_identity":
                    iz += w * r.fisher_joint
    return {n: float(np.trace(v)) for n, v in tot.items()}, float(np.trace(iz)), ok


def steps_from_schedule(sched, cap=None):
    T = S.HORIZON
    out = []
    for t in range(T):
        counts = {}
        for traj in sched:
            counts[tuple(traj[t])] = counts.get(tuple(traj[t]), 0) + 1
        items = sorted(counts.items(), key=lambda kv: -kv[1])
        if cap:
            items = items[:cap]
        z = sum(c for _, c in items)
        out.append([(a, c / z) for a, c in items])
    return out


# ---------------- evaluator (same for every rule) ----------------
def simulate(model, rng, n_traj, sched, states=None):
    """True (tau, cells) + per-sensor sampled category indices [m,t,k]."""
    T, k = S.HORIZON, len(model.sensors)
    Tn, Ta = model.transition("n", model.slip0), model.transition("a", model.slip0)
    taus, cells = np.zeros(n_traj, int), np.zeros((n_traj, T), int)
    for m in range(n_traj):
        if states is not None:
            taus[m] = states[m][1][1]
            cells[m] = [model.idx[tuple(s[0])] for s in states[m][1:T + 1]]
            continue
        taus[m] = rng.random() < 0.5
        c = rng.choice(model.n, p=model.init_cells)
        P = Ta if taus[m] else Tn
        for t in range(T):
            if t:
                c = rng.choice(model.n, p=P[c])
            cells[m, t] = c
    cats = np.zeros((n_traj, T, k), int)
    for m in range(n_traj):
        for t in range(T):
            for j in range(k):
                row = model.sensor_cat_matrix(j, sched[m][t][j])[cells[m, t]]
                cats[m, t, j] = rng.choice(4, p=row)
    return taus, cells, cats


def evaluate_rule(model, rule, sched, taus, cats, cache):
    T, k = S.HORIZON, len(model.sensors)
    tuples = S.all_tuples(k)
    Tn, Ta = model.transition("n", model.slip0), model.transition("a", model.slip0)
    H, correct = [], 0
    for m in range(len(taus)):
        b = np.stack([0.5 * model.init_cells, 0.5 * model.init_cells])
        for t in range(T):
            acts = tuple(sched[m][t])
            key = (rule.__name__, acts)
            if key not in cache:
                L = model.tuple_likelihood(acts)
                labels = [rule(z) for z in tuples]
                uniq = sorted(set(labels))
                li = np.array([uniq.index(x) for x in labels])
                agg = np.zeros((model.n, len(uniq)))
                for c in range(len(uniq)):
                    agg[:, c] = L[:, li == c].sum(axis=1)
                cache[key] = (agg, {u: i for i, u in enumerate(uniq)})
            agg, lab_idx = cache[key]
            z = tuple(S.CATS[c] for c in cats[m, t])
            if t:
                b = np.stack([b[0] @ Tn, b[1] @ Ta])
            b = b * agg[:, lab_idx[rule(z)]][None, :]
            b /= b.sum()
        p1 = b[1].sum()
        H.append(-sum(q * np.log2(q) for q in (p1, 1 - p1) if q > 0))
        correct += int((p1 > 0.5) == bool(taus[m]))
    return float(np.mean(H)), correct / len(taus)


# ---------------- main experiment ----------------
def run_config(policy_type, dynamics, sensor_range, maps, retrain=None, training_time=0.0):
    model = S.GridModel(dynamics, sensor_range, S.IMAS2_SELECTED)
    rng = np.random.default_rng(SEED)
    n = len(retrain["acts"]) if retrain else N_TRAJ
    sched = action_schedule(model, policy_type, rng, n, retrain)
    t0 = time.time()
    steps = steps_from_schedule(sched, cap=N_RANDOM_ACTS if policy_type != "fixed_orientation" else None)
    losses, tr_iz, ok = fisher_by_rule(model, steps, maps)
    sel_t = time.time() - t0
    taus, _, cats = simulate(model, np.random.default_rng(SEED + 1), n, sched,
                             states=retrain["states"] if retrain else None)
    cache, rows = {}, []
    for name, rule, _ in S.RULE_SPECS:
        h, acc = evaluate_rule(model, rule, sched, taus, cats, cache)
        rows.append({"rule": name, "policy_type": policy_type, "dynamics": dynamics,
                     "sensor_range": sensor_range, "tr_delta_I": losses[name], "cond_entropy": h,
                     "accuracy": acc, "seed": SEED, "n_trajectories": n,
                     "selection_time_s": sel_t, "training_time_s": training_time,
                     "tr_I_Z": tr_iz, "status": "ok" if ok else "check"})
    return rows


def rank_agreement(rows):
    d = np.array([r["tr_delta_I"] for r in rows])
    h = np.array([r["cond_entropy"] for r in rows])
    a = np.array([r["accuracy"] for r in rows])
    return {"spearman_dI_vs_entropy": float(spearmanr(d, h)[0]),
            "spearman_dI_vs_neg_accuracy": float(spearmanr(d, -a)[0])}


# ---------------- placements ----------------
def placement_scores(dynamics, sensor_range="small"):
    rows = []
    near = [m for m in S.build_rule_maps() if m[0] == "near_identity"]
    fisher_t = mi_t = 0.0
    for pl in combinations(range(9), 5):
        model = S.GridModel(dynamics, sensor_range, pl)
        acts = fixed_orientation(model)
        t0 = time.time()
        steps = [[(acts, 1.0)] for _ in range(S.HORIZON)]
        _, tr_iz, _ = fisher_by_rule(model, steps, near)
        fisher_t += time.time() - t0
        t0 = time.time()
        sched = [[acts] * S.HORIZON for _ in range(N_TRAJ)]
        taus, _, cats = simulate(model, np.random.default_rng(SEED + 1), N_TRAJ, sched)
        h, acc = evaluate_rule(model, S.rule_near_identity, sched, taus, cats, {})
        mi_t += time.time() - t0
        rows.append({"placement": "-".join(map(str, pl)), "fisher_score": tr_iz, "mi_score": 1.0 - h,
                     "cond_entropy": h, "accuracy": acc, "dynamics": dynamics,
                     "sensor_range": sensor_range, "orientation": "fixed_orientation"})
    f = np.array([r["fisher_score"] for r in rows])
    mi = np.array([r["mi_score"] for r in rows])
    top_f = set(np.argsort(-f)[:5])
    top_mi = set(np.argsort(-mi)[:5])
    pick = lambda idx: {"placement": rows[idx]["placement"], "cond_entropy": rows[idx]["cond_entropy"],
                        "accuracy": rows[idx]["accuracy"]}
    imas = "-".join(map(str, S.IMAS2_SELECTED))
    imas_i = next(i for i, r in enumerate(rows) if r["placement"] == imas)
    summary = {
        "dynamics": dynamics, "sensor_range": sensor_range, "n_placements": len(rows),
        "spearman": float(spearmanr(f, mi)[0]), "top5_overlap": len(top_f & top_mi),
        "fisher_top1": pick(int(np.argmax(f))), "mi_top1": pick(int(np.argmax(mi))),
        "fisher_top5": [rows[i]["placement"] for i in sorted(top_f, key=lambda i: -f[i])],
        "mi_top5": [rows[i]["placement"] for i in sorted(top_mi, key=lambda i: -mi[i])],
        "imas2_reported_pick": {**pick(imas_i), "fisher_rank": int((f > f[imas_i]).sum()) + 1,
                                "mi_rank": int((mi > mi[imas_i]).sum()) + 1},
        "selection_time_s": {"fisher_all_126": fisher_t, "mi_mc_all_126": mi_t},
        "training_time_s": 0.0,
        "mi_estimator": f"Monte-Carlo 1 - mean H(tau|y), M={N_TRAJ}, seed={SEED + 1}, near_identity fusion",
        "fisher_estimator": "Σ_t tr I_Z(θ=(prior_logit,slip)) of 5-sensor categorical report",
    }
    return rows, summary


def write_csv(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    maps = S.build_rule_maps()
    main_rows, agreement = [], []
    for pt in ("fixed_orientation", "random_orientation"):
        for dyn in ("deterministic", "stochastic"):
            for rng_ in ("small", "large"):
                rows = run_config(pt, dyn, rng_, maps)
                main_rows += rows
                agreement.append({"policy_type": pt, "dynamics": dyn, "sensor_range": rng_, **rank_agreement(rows)})
                print(pt, dyn, rng_, agreement[-1], flush=True)
    retrain_meta = None
    if RETRAIN_JSON.exists():
        rj = json.loads(RETRAIN_JSON.read_text())
        T = S.HORIZON
        primary_first = rj["order_primary_first"]
        perm = [primary_first.index(s) for s in S.IMAS2_SELECTED]
        retrain = {"acts": [[tuple(tr["acts"][t][p] for p in perm) for t in range(T)] for tr in rj["trajectories"]],
                   "states": [tr["states"] for tr in rj["trajectories"]]}
        tt = float(sum(rj["training_time_s"].values()))
        rows = run_config("retrained_imas2", "stochastic", "small", maps, retrain=retrain, training_time=tt)
        main_rows += rows
        agreement.append({"policy_type": "retrained_imas2", "dynamics": "stochastic", "sensor_range": "small",
                          **rank_agreement(rows)})
        retrain_meta = {k: rj[k] for k in ("order_primary_first", "iters", "M", "T", "training_time_s",
                                            "rollout_time_s", "final_entropy_last50")}
    write_csv(DATA / "imas2_rule_comparison.csv", main_rows, MAIN_FIELDS)

    place_det, sum_det = placement_scores("deterministic")
    place_sto, sum_sto = placement_scores("stochastic")
    write_csv(DATA / "imas2_placements.csv", place_det)
    write_csv(DATA / "imas2_placements_stochastic.csv", place_sto)

    reported = {
        "source": "AAMAS_2026.pdf in upstream repo / arXiv:2510.20009 (REPORTED, not rerun)",
        "selected_sensors_det_small": [2, 3, 4, 6, 8],
        "final_cond_entropy_det_small_5sensors": 0.367, "accuracy_det_small": 0.88,
        "table1_stochastic_large": {"IMAS2": {"entropy": 0.493, "accuracy": 0.860, "time_per_iter_s": 1.58},
                                    "IPG_fixed": {"entropy": 0.525, "accuracy": 0.755, "time_per_iter_s": 7.62},
                                    "IPG_random": {"entropy": 0.558, "accuracy": 0.707, "time_per_iter_s": 7.63},
                                    "IPG_visibility": {"entropy": 0.502, "accuracy": 0.841, "time_per_iter_s": 7.63}},
        "training_setup": "M=100, T=15, N=1000 iters/sensor, lr=1e-3, ~1.3 s/iter (i7-12700)",
    }
    summary = {
        "upstream": {"repo": "https://github.com/AronYoung414/multi-agent-active-perception-grid-world",
                     "commit": "0eae15604c39efac4d8fe4e5e56fbe20aac5d732", "license": "MIT"},
        "theta": list(S.PARAMETERS), "horizon_sensing_steps": S.HORIZON, "obs_noise": S.OBS_NOISE,
        "placement_main": list(S.IMAS2_SELECTED), "seed": SEED, "n_trajectories": N_TRAJ,
        "rank_agreement": agreement,
        "placements": {"deterministic_small": sum_det, "stochastic_small": sum_sto},
        "retrained_imas2": retrain_meta or "not available (see blockers)",
        "reported_imas2": reported,
        "caveats": [
            "Fisher on θ=(prior_logit, slip) is a local-sensitivity surrogate, not mutual information.",
            "ΔI is a time-summed per-step snapshot Fisher (ignores cross-step correlation).",
            "Deterministic dynamics: slip derivative by forward FD at slip=0 (boundary).",
            "Per-sensor categories adv/nom/unc from fixed centre occupancy LR (θ-independent).",
            "Evaluator: exact HMM forward filter on fused labels, same seeds/trajectories for every rule.",
            "Retrained rows: reduced IMAS² (fixed reported order, fewer iters); obs resampled in our evaluator given logged states/actions.",
            "Reported IMAS² numbers are copied from the paper and never mixed with rerun numbers.",
        ],
    }
    (DATA / "imas2_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"placements_det": {k: sum_det[k] for k in ("spearman", "top5_overlap")},
                      "placements_sto": {k: sum_sto[k] for k in ("spearman", "top5_overlap")}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
