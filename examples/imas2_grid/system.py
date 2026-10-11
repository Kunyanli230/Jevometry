"""IMAS² grid-world twin for coord-fisher AggregationMaps.

Reuses the vendored IMAS² environment (third_party/imas2_grid, MIT) for the
10x10 grid, obstacles, 9 candidate sensors and the value-iteration robot
policies (nominal goal (9,7), adversary goal (9,2)).  Sensors follow the IMAS²
observation model: if the robot is inside the chosen NW/NE/SW/SE footprint the
sensor returns its exact cell w.p. 1-obs_noise, else null.

Fisher surrogate (NOT mutual information): θ = (prior_logit, slip) where
prior_logit = logit P(type=adversary) and slip is the motion slip probability.
At every step t the fused report Z_t (5 per-sensor categories) has the exact
mixture law P_t(z;θ) = Σ_s μ_t(s;θ) Π_i P(z_i | s, a_i); we sum the per-step
Fisher matrices over the horizon (time-summed snapshot Fisher, ignores
cross-step correlation).  This is a local-sensitivity surrogate.

Per-sensor categories (θ-independent, fixed at the centre):
  adv / nom  — detected cell whose centre occupancy likelihood ratio favours one type
  unc        — detected but ambiguous cell (LR in [1/3, 3])  → quarantine target
  none       — null observation
"""

from __future__ import annotations

import sys
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "imas2_grid"))

from grid_world_example import Environment  # noqa: E402  (vendored, MIT)

from jevometry.schemas.joint import ConstructionMode, JointDistribution  # noqa: E402
from jevometry.systems.pushforward import AggregationMap  # noqa: E402

PARAMETERS = ("prior_logit", "slip")
HORIZON = 14  # sensing steps (IMAS² T=15 incl. final end action)
OBS_NOISE = 0.5
IMAS2_SELECTED = (2, 3, 4, 6, 8)  # reported IMAS² pick (det., small range)
CATS = ("adv", "nom", "unc", "none")
ACTIONS = ("NW", "NE", "SW", "SE")
RANGE_OFFSETS = {
    "small": {
        "NW": [(0, 0), (-1, 0), (-1, 1), (0, 1)],
        "NE": [(0, 0), (1, 0), (1, 1), (0, 1)],
        "SW": [(0, 0), (-1, 0), (-1, -1), (0, -1)],
        "SE": [(0, 0), (1, 0), (1, -1), (0, -1)],
    },
    "large": {
        "NW": [(0, 0), (-1, 0), (-1, 1), (0, 1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2)],
        "NE": [(0, 0), (1, 0), (1, 1), (0, 1), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)],
        "SW": [(0, 0), (-1, 0), (-1, -1), (0, -1), (-2, 0), (-2, -1), (-2, -2), (-1, -2), (0, -2)],
        "SE": [(0, 0), (1, 0), (1, -1), (0, -1), (2, 0), (2, -1), (2, -2), (1, -2), (0, -2)],
    },
}
CORE_RULE_NAMES = ("near_identity", "wider", "baseline", "stricter")


@lru_cache(maxsize=4)
def build_env(slip: float) -> Environment:
    env = Environment()
    if abs(env.stoPar - slip) > 1e-12:
        env.stoPar = slip
        env.gr_transition = env.get_transition()
        env.opt_policy_n = env.value_iteration(env.rewards_n)[1]
        env.opt_policy_a = env.value_iteration(env.rewards_a)[1]
    return env


class GridModel:
    """Vectorised IMAS² dynamics + emissions for one (dynamics, range) config."""

    def __init__(self, dynamics: str, sensor_range: str, sensors: tuple[int, ...]):
        self.dynamics = dynamics
        self.slip0 = 0.0 if dynamics == "deterministic" else 0.1
        self.env = build_env(self.slip0)
        self.cells = sorted(self.env.gr_states)
        self.idx = {c: i for i, c in enumerate(self.cells)}
        self.n = len(self.cells)
        self.sensor_range = sensor_range
        self.sensors = tuple(sensors)
        self.positions = [self.env.sensors[k] for k in self.sensors]
        self.policy = {
            "n": {s: max(p, key=p.get) for s, p in self.env.opt_policy_n.items()},
            "a": {s: max(p, key=p.get) for s, p in self.env.opt_policy_a.items()},
        }
        self.init_cells = np.zeros(self.n)
        for c in self.env.gr_initial_states:
            self.init_cells[self.idx[c]] = 1.0 / len(self.env.gr_initial_states)
        # coverage[k][act] -> boolean vector over cells, for ALL 9 candidates
        self.coverage_all = [
            {a: self._coverage(pos, a) for a in ACTIONS} for pos in self.env.sensors
        ]
        self.coverage = [self.coverage_all[k] for k in self.sensors]
        self.theta0 = np.array([0.0, self.slip0])
        self.cell_class = self._classify_cells()

    # ---------- dynamics ----------
    def _coverage(self, pos, act) -> np.ndarray:
        v = np.zeros(self.n, dtype=bool)
        for dx, dy in RANGE_OFFSETS[self.sensor_range][act]:
            c = (pos[0] + dx, pos[1] + dy)
            if c in self.idx:
                v[self.idx[c]] = True
        return v

    def transition(self, kind: str, slip: float) -> np.ndarray:
        """Row-stochastic cell transition under the fixed centre policy."""
        T = np.zeros((self.n, self.n))
        for s in self.cells:
            i = self.idx[s]
            act = self.policy[kind][s]
            if act == (0, 0):
                T[i, i] = 1.0
                continue
            main = (s[0] + act[0], s[1] + act[1])
            T[self.idx[main] if main in self.idx else i, i] += 0.0
            T[i, self.idx[main] if main in self.idx else i] += 1 - 2 * slip
            comp = [(1, 0), (-1, 0)] if act[0] == 0 else [(0, 1), (0, -1)]
            for d in comp:
                c = (s[0] + d[0], s[1] + d[1])
                T[i, self.idx[c] if c in self.idx else i] += slip
        return T

    def occupancy(self, theta: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
        """μ[t, tau, cell] for t=0..horizon-1 (tau 0 nominal, 1 adversary)."""
        p_adv = 1.0 / (1.0 + np.exp(-theta[0]))
        Tn, Ta = self.transition("n", theta[1]), self.transition("a", theta[1])
        mu = np.zeros((horizon, 2, self.n))
        mu[0, 0], mu[0, 1] = (1 - p_adv) * self.init_cells, p_adv * self.init_cells
        for t in range(1, horizon):
            mu[t, 0] = mu[t - 1, 0] @ Tn
            mu[t, 1] = mu[t - 1, 1] @ Ta
        return mu

    def _classify_cells(self) -> np.ndarray:
        """0=adv,1=nom,2=unc per cell from centre occupancy LR (θ-independent)."""
        mu = self.occupancy(self.theta0).sum(axis=0)
        eps = 1e-9
        lr = (mu[1] + eps) / (mu[0] + eps)
        cls = np.full(self.n, 2)
        cls[lr > 3.0] = 0
        cls[lr < 1.0 / 3.0] = 1
        return cls

    # ---------- emissions ----------
    def sensor_cat_matrix(self, j: int, act: str) -> np.ndarray:
        """P(cat | cell) for selected sensor j (n × 4)."""
        cov = self.coverage[j][act]
        M = np.zeros((self.n, 4))
        M[:, 3] = 1.0
        rows = np.where(cov)[0]
        M[rows, 3] = OBS_NOISE
        M[rows, self.cell_class[rows]] = 1 - OBS_NOISE
        return M

    def tuple_likelihood(self, acts: tuple[str, ...]) -> np.ndarray:
        """P(z | cell) over all 4^k category tuples (n × 4^k), product order."""
        L = np.ones((self.n, 1))
        for j, a in enumerate(acts):
            L = (L[:, :, None] * self.sensor_cat_matrix(j, a)[:, None, :]).reshape(self.n, -1)
        return L


def all_tuples(k: int) -> list[tuple[str, ...]]:
    return list(product(CATS, repeat=k))


# ---------------- coordination rules over 5 per-sensor categories ----------------
def _counts(z):
    return z.count("adv"), z.count("nom"), z.count("unc")


def _majority(na, nn):
    if na == 0 and nn == 0:
        return "none"
    return "adv" if na > nn else ("nom" if nn > na else "tie")


def rule_near_identity(z):
    return "|".join(z)


def rule_baseline(z):
    na, nn, nu = _counts(z)
    return "reject" if nu else _majority(na, nn)


def rule_stricter(z):
    na, nn, nu = _counts(z)
    if nu or (na and nn):
        return "reject"
    if na >= 2:
        return "adv"
    if nn >= 1:
        return "nom"
    return "none"  # single adv or nothing


def rule_wider(z):
    na, nn, nu = _counts(z)
    return "reject" if nu else f"a{min(na, 2)}n{min(nn, 2)}"


def rule_quarantine_review(z):
    na, nn, _ = _counts(z)  # uncertain sensors isolated (dropped), not rejected
    return _majority(na, nn)


def rule_quarantine_wider(z):
    na, nn, _ = _counts(z)
    return f"a{min(na, 2)}n{min(nn, 2)}"


def rule_coarse_binary(z):
    na, _, nu = _counts(z)
    return "reject" if nu else ("alarm" if na else "clear")


RULE_SPECS = [
    ("baseline", rule_baseline, "uncertain→reject; else majority adv/nom/tie/none"),
    ("stricter", rule_stricter, "uncertain or conflict→reject; adv needs ≥2"),
    ("wider", rule_wider, "uncertain→reject; else capped (n_adv,n_nom) counts"),
    ("quarantine_review", rule_quarantine_review, "isolate uncertain sensors; majority"),
    ("quarantine_wider", rule_quarantine_wider, "isolate uncertain sensors; capped counts"),
    ("coarse_binary", rule_coarse_binary, "uncertain→reject; else alarm vs clear"),
    ("near_identity", rule_near_identity, "one label per joint report (upper bound)"),
]
RULES = {n: f for n, f, _ in RULE_SPECS}


def build_rule_maps(k: int = 5) -> list[tuple[str, AggregationMap]]:
    tuples = all_tuples(k)
    return [
        (n, AggregationMap(name=n, mapping={z: f(z) for z in tuples}, theta_dependent=False, description=d))
        for n, f, d in RULE_SPECS
    ]


def step_joint(model: GridModel, theta: np.ndarray, t: int, acts: tuple[str, ...]) -> np.ndarray:
    mu = model.occupancy(theta, horizon=t + 1)[t].sum(axis=0)
    return mu @ model.tuple_likelihood(acts)


def step_joint_and_jacobian(model, t, acts, h=1e-5):
    """Exact mixture law + finite-difference Jacobian (forward FD on slip at 0)."""
    th0 = model.theta0
    p0 = step_joint(model, th0, t, acts)
    jac = np.zeros((p0.size, 2))
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        if k == 1 and th0[1] - h < 0:
            jac[:, k] = (step_joint(model, th0 + e, t, acts) - p0) / h
        else:
            jac[:, k] = (step_joint(model, th0 + e, t, acts) - step_joint(model, th0 - e, t, acts)) / (2 * h)
    return p0, jac


def joint_distribution(p: np.ndarray, jac: np.ndarray, k: int, tol: float = 1e-12):
    tuples = all_tuples(k)
    keep = np.where(p > tol)[0]
    pk = p[keep] / p[keep].sum()
    jd = JointDistribution(
        mode=ConstructionMode.EXPLICIT,
        node_order=[f"sensor_{i}" for i in range(k)],
        outcomes=[tuples[i] for i in keep],
        probabilities=pk.tolist(),
        assumptions=["imas2 grid: exact mixture over robot cell at step t"],
    )
    return jd, jac[keep]
