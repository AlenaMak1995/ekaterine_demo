# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Set, List
from dataclasses import field
import cvxpy as cp
import numpy as np
import time
import matplotlib.pyplot as plt


from ..src.gridworld import GridWorld, State, Action

AugState = tuple[Any, ...]
Region = Set[State]
PolicyAug = dict[AugState, dict[Action, float]]
XOptAug = dict[tuple[AugState, Action], float]

@dataclass
class RegionFlagSpec:
    # this flag becomes 1 if we ever visit `region` - reachability PCTL constraint
    name: str
    region: Region


@dataclass
class PCTLRegionConstraint:
    # reachability-style constraints using flags
    # P(ever visit region_i) <= bound  -> kind = 'visit_region_max'
    # P(ever visit region_i) >= bound  -> kind = 'visit_region_min'
    
    kind: str
    region_name: str
    bound: float


@dataclass
class UntilSpec:
    # for A U B, A_region = states where A holds, B_region = states where B holds
    # the formula succeeds if you hit B while A has held at all previous steps; fails if you ever leave A before hitting B.
    name: str
    A_region: Region
    B_region: Region


@dataclass
class UntilConstraint:

    # constraints for until
    # kind = "until_min":  P( A U B ) >= bound
    # kind = "until_max":  P( A U B ) <= bound

    kind: str  
    spec_name: str
    bound: float

@dataclass
class AugmentedMDPBaseline:

    # augmented MDP = base GridWorld × bits:
    # first len(flags) bits: "have we visited region i?"
    # for each UntilSpec name: 2 bits (success, fail)[success = 1  if A U B has been satisfied, fail    = 1  if A U B has been violated]

    base: GridWorld
    flags: List[RegionFlagSpec]
    until_specs: List[UntilSpec] = field(default_factory=list)

    def __post_init__(self):
        # indices for simple region flags
        self.flag_indices = {f.name: idx for idx, f in enumerate(self.flags)}

        # indices for until success/fail bits
        self.until_armed_idx: Dict[str, int] = {}
        self.until_success_idx: Dict[str, int] = {}
        self.until_fail_idx: Dict[str, int] = {}

        offset = len(self.flags)
        for i, uspec in enumerate(self.until_specs):
          base = offset + 3 * i
          self.until_armed_idx[uspec.name]   = base + 0
          self.until_success_idx[uspec.name] = base + 1
          self.until_fail_idx[uspec.name]    = base + 2

        total_bits = len(self.flags) + 3 * len(self.until_specs)
        self.states_aug: List[AugState] = []

        # all 0/1 combinations for all bits
        if total_bits > 0:
            Z_space = list(np.ndindex(*(2 for _ in range(total_bits))))
        else:
            Z_space = [()]

        for s in self.base.states:
            for z in Z_space:
                self.states_aug.append((s,) + tuple(z))

    def is_absorbing_aug(self, st: AugState) -> bool:
        # Absorb ONLY at base goal.
        # Do NOT absorb on flags
      return self.base.is_goal(st[0])

    def actions_from_aug(self, st: AugState) -> List[Action]:
      if self.is_absorbing_aug(st):
        return []
      return self.base.actions_from(st[0])

    def _update_bits_from_next_state(self, bits: List[int], s2: State) -> List[int]:


          # simple "visited region" flags (monotone: 0 -> 1 and stays 1)
          for i, spec in enumerate(self.flags):
            if s2 in spec.region:
                bits[i] = 1

          # A U B formula bits (success/fail, also monotone)
          for uspec in self.until_specs:
            i_arm  = self.until_armed_idx[uspec.name]
            i_succ = self.until_success_idx[uspec.name]
            i_fail = self.until_fail_idx[uspec.name]

            # already decided for this formula?
            if bits[i_succ] == 1 or bits[i_fail] == 1:
                continue

            if bits[i_arm] == 0:
                # strict until: if it was not already true/armed from the start,
                # it cannot become true later by "entering A"
                bits[i_fail] = 1
                continue
            # Armed: now enforce staying in A until hitting B
            if s2 in uspec.B_region:
              bits[i_succ] = 1
              bits[i_fail] = 0
            elif s2 not in uspec.A_region:
              bits[i_fail] = 1
              bits[i_succ] = 0        

          return bits

    def move_aug(self, st: AugState, a: Action) -> AugState:
          # Deterministic next-state (kept for debugging).

          s = st[0]
          bits = list(st[1:])
          s2 = self.base.move(s, a) 
          bits2 = self._update_bits_from_next_state(bits, s2)
          # if we reached the base goal and this until never succeeded, mark failure
          if self.base.is_goal(s2):
              for uspec in self.until_specs:
                  i_succ = self.until_success_idx[uspec.name]
                  i_fail = self.until_fail_idx[uspec.name]
                  if bits2[i_succ] == 0:
                      bits2[i_fail] = 1
          return (s2, *bits2)

    def transitions_aug(self, st: AugState, a: Action) -> Dict[AugState, float]:
          # Stochastic augmented transition kernel.

          s = st[0]
          bits0 = list(st[1:])
          dist: Dict[AugState, float] = {}

          base_dist = self.base.transitions(s, a)  # {s2: p}
          for s2, p in base_dist.items():
              bits2 = self._update_bits_from_next_state(bits0.copy(), s2)

              if self.base.is_goal(s2):
                  for uspec in self.until_specs:
                      i_succ = self.until_success_idx[uspec.name]
                      i_fail = self.until_fail_idx[uspec.name]
                      if bits2[i_succ] == 0:
                          bits2[i_fail] = 1

              st2 = (s2, *bits2)
              dist[st2] = dist.get(st2, 0.0) + float(p)  

          # numeric safety
          total = sum(dist.values())
          if total > 0 and abs(total - 1.0) > 1e-12:
            for k in list(dist.keys()):
                dist[k] /= total

          return dist


    def cost_aug(self, st: AugState, a: Action) -> float:
        return self.base.cost(st[0], a)

    @property
    def start_aug(self) -> AugState:
        total_bits = len(self.flags) + 3 * len(self.until_specs)
        bits = [0 for _ in range(total_bits)]

        s0 = self.base.start

        # initialize simple region flags at start
        for i, spec in enumerate(self.flags):
            if s0 in spec.region:
                bits[i] = 1

        # initialize strict-until status at start
        for uspec in self.until_specs:
            i_arm  = self.until_armed_idx[uspec.name]
            i_succ = self.until_success_idx[uspec.name]
            i_fail = self.until_fail_idx[uspec.name]

            if s0 in uspec.B_region:
                bits[i_succ] = 1
            elif s0 in uspec.A_region:
                bits[i_arm] = 1
            else:
                bits[i_fail] = 1

        return (self.base.start,) + tuple(bits)


def solve_lp_with_pctl_aug_baseline(
    mdp_aug: AugmentedMDPBaseline,
    p_goal_min: float,
    region_constraints: List[PCTLRegionConstraint],
    until_constraints: List[UntilConstraint],
) -> tuple[
    float,
    float,
    XOptAug,
    Dict[str, float],
    Dict[str, float],
    float,
]:

    # global LP over augmented MDP with: P(reach goal) >= p_goal_min, region constraints on P(ever visit region) for monotone visit-flags, until constraints on P(A U B) for each UntilSpec

    reachable: Set[AugState] = set()
    frontier: List[AugState] = [mdp_aug.start_aug]

    while frontier:
        st = frontier.pop()
        if st in reachable:
            continue

        reachable.add(st)

        if mdp_aug.is_absorbing_aug(st):
            continue

        for a in mdp_aug.actions_from_aug(st):
            for st2 in mdp_aug.transitions_aug(st, a):
                if st2 not in reachable:
                    frontier.append(st2)

    non_abs_states: List[AugState] = [
        st for st in reachable
        if not mdp_aug.is_absorbing_aug(st)
    ]

    state_index: Dict[AugState, int] = {
        st: i for i, st in enumerate(non_abs_states)
    }

    edges: List[Tuple[AugState, Action]] = []
    for st in non_abs_states:
        for a in mdp_aug.actions_from_aug(st):
            edges.append((st, a))

    E = len(edges)
    if E == 0:
        return 0.0, 0.0, {}, {}, {}, 0.0

    x_vec = cp.Variable(E, nonneg=True)

    # Flow constraints (stochastic) (out(st) - sum_{prev, a} P(st | prev,a) x(prev,a) = 1{st=start})
    S = len(non_abs_states)
    A_flow = np.zeros((S, E))
    b = np.zeros(S)

    for e, (st, a) in enumerate(edges):
        i_from = state_index[st]
        A_flow[i_from, e] += 1.0  # outflow from st

        for st2, p in mdp_aug.transitions_aug(st, a).items():
            if (not mdp_aug.is_absorbing_aug(st2)) and (st2 in state_index):
                i_to = state_index[st2]
                A_flow[i_to, e] -= float(p)

    start_idx = state_index.get(mdp_aug.start_aug, None)
    if start_idx is not None:
        b[start_idx] = 1.0

    constraints: List[cp.Constraint] = [A_flow @ x_vec == b]

    # probability coefficient vectors
    goal_coeff = np.zeros(E)

    region_coeffs: Dict[str, np.ndarray] = {
        spec.name: np.zeros(E) for spec in mdp_aug.flags
    }
    until_coeffs: Dict[str, np.ndarray] = {
        uspec.name: np.zeros(E) for uspec in mdp_aug.until_specs
    }

    for e, (st, a) in enumerate(edges):
        bits = st[1:]  

        for st2, p in mdp_aug.transitions_aug(st, a).items():
            s2 = st2[0]
            bits2 = st2[1:]
            p = float(p)

            # reach-goal probability: probability mass on edges that enter goal
            if s2 in mdp_aug.base.goal:
                goal_coeff[e] += p

            # ever-visit region_i: count the *first* time bit flips 0->1
            for spec in mdp_aug.flags:
                idx = mdp_aug.flag_indices[spec.name]
                if bits[idx] == 0 and bits2[idx] == 1:
                    region_coeffs[spec.name][e] += p

            # until success probability: count the transition that triggers success first time
            for uspec in mdp_aug.until_specs:
                i_arm  = mdp_aug.until_armed_idx[uspec.name]
                i_succ = mdp_aug.until_success_idx[uspec.name]
                i_fail = mdp_aug.until_fail_idx[uspec.name]

                if (
                    bits[i_succ] == 0
                    and bits[i_fail] == 0
                    and bits[i_arm] == 1
                    and bits2[i_succ] == 1
                    and bits2[i_fail] == 0
                ):
                    until_coeffs[uspec.name][e] += p
    debug_name = "G2U_G3"
    # if debug_name in until_coeffs:
    #     print("\n[CVXPY DEBUG]")
    #     print("until_coeff nnz:", int(np.count_nonzero(until_coeffs[debug_name])))
    #     print("until_coeff sum :", float(until_coeffs[debug_name].sum()))
    #     print("until_coeff max :", float(until_coeffs[debug_name].max()))

    goal_prob = goal_coeff @ x_vec
    region_prob = {}

    for spec in mdp_aug.flags:
        idx = mdp_aug.flag_indices[spec.name]
        init_val = float(mdp_aug.start_aug[1 + idx] == 1)
        region_prob[spec.name] = init_val + region_coeffs[spec.name] @ x_vec
    until_prob  = {name: coeff @ x_vec for name, coeff in until_coeffs.items()}

    # add constraints

    constraints.append(goal_prob >= float(p_goal_min))

    for rc in region_constraints:
        expr = region_prob[rc.region_name]
        if rc.kind == "visit_region_max":
            constraints.append(expr <= float(rc.bound))
        elif rc.kind == "visit_region_min":
            constraints.append(expr >= float(rc.bound))
        else:
            raise ValueError(f"Unknown region constraint kind='{rc.kind}'")

    for uc in until_constraints:
        expr = until_prob[uc.spec_name]
        if uc.kind == "until_min":
            constraints.append(expr >= float(uc.bound))
        elif uc.kind == "until_max":
            constraints.append(expr <= float(uc.bound))
        else:
            raise ValueError(f"Unknown until constraint kind='{uc.kind}'")

    # Objective: minimize expected cumulative cost
    c_vec = np.zeros(E)
    for e, (st, a) in enumerate(edges):
        c_vec[e] = mdp_aug.cost_aug(st, a)

    objective = cp.Minimize(c_vec @ x_vec )
    prob = cp.Problem(objective, constraints)

    t0 = time.perf_counter()
    # print("Until constraints:", [(uc.kind, uc.spec_name, uc.bound) for uc in until_constraints])
    # print("Has key G2U_G3 in until_prob?", "G2U_G3" in until_prob)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    t1 = time.perf_counter()

    if prob.status not in ("optimal", "optimal_inaccurate"):
        # infeasible/other → keep return shape consistent
        return float("inf"), 0.0, {}, {k: 0.0 for k in region_prob}, {k: 0.0 for k in until_prob}, (t1 - t0)

    # line around coeff/value extraction
    x_opt_raw = x_vec.value
    if x_opt_raw is None:
        return float("inf"), 0.0, {}, {k: 0.0 for k in region_prob}, {k: 0.0 for k in until_prob}, (t1 - t0)

    x_opt = np.asarray(x_opt_raw, dtype=float)

    x_opt = np.asarray(x_opt_raw, dtype=float)
    x_opt_dict: XOptAug = {
        edges[i]: float(x_opt[i])
        for i in range(E)
        if x_opt[i] > 1e-12
    }

    J = float((c_vec @ x_vec).value)
    p_goal_out = float(goal_prob.value)
    region_probs_out = {k: float(v.value) for k, v in region_prob.items()}
    until_probs_out  = {k: float(v.value) for k, v in until_prob.items()}

    return J, p_goal_out, x_opt_dict, region_probs_out, until_probs_out, (t1 - t0)



def recover_policy_from_x_aug(
    mdp_aug: AugmentedMDPBaseline,
    x_opt: XOptAug,
    tol: float = 1e-8,
) -> PolicyAug:
    policy = {}
    for st in mdp_aug.states_aug:
        if mdp_aug.is_absorbing_aug(st):
            continue
        flows = [(a, x_opt.get((st, a), 0.0)) for a in mdp_aug.actions_from_aug(st)]
        total = sum(v for _, v in flows)
        if total > tol:
            policy[st] = {a: v / total for a, v in flows}
        else:
            policy[st] = {a: 0.0 for a in mdp_aug.actions_from_aug(st)}
    return policy


def print_policy_grid_z0(
    mdp_aug: AugmentedMDPBaseline,
    policy_aug: PolicyAug,
) -> None:
    arrow = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
    total_bits = len(mdp_aug.flags) + 3 * len(mdp_aug.until_specs)

    for r in range(mdp_aug.base.N):
        row = ""
        for c in range(mdp_aug.base.N):
            s = (r, c)
            if s == mdp_aug.base.start:
                st0 = mdp_aug.start_aug
            else:
                st0 = (s,) + tuple(0 for _ in range(total_bits))
            if s in mdp_aug.base.goal:
                row += " G  "
            elif st0 not in policy_aug:
                row += " ·  "
            else:
                probs = policy_aug[st0]
                if not probs:
                    row += " ·  "
                    continue
                best_a = max(probs, key=lambda a: probs[a])
                if probs[best_a] < 1e-6:
                    row += " ·  "
                else:
                    row += f" {arrow[best_a]}  "
        print(row)


def simulate_policy_aug(
    mdp_aug: AugmentedMDPBaseline,
    policy_aug: PolicyAug,
    max_steps: int = 100,
    seed: int = 0,
    greedy: bool = False,
) -> tuple[List[State], List[AugState]]:
    rng = np.random.default_rng(seed)

    st = mdp_aug.start_aug
    base_traj = [st[0]]
    aug_traj  = [st]

    for t in range(max_steps):
        if mdp_aug.is_absorbing_aug(st):
            break

        probs = policy_aug.get(st, None)
        if probs is None:
            break

        actions = list(probs.keys())
        p = np.array([probs[a] for a in actions], dtype=float)
        if p.sum() <= 1e-12:
            p = np.ones(len(actions)) / len(actions)
        else:
            p /= p.sum()

        if greedy:
            a = actions[int(np.argmax(p))]
        else:
            i = rng.choice(len(actions), p=p)
            a = actions[i]

        dist = mdp_aug.transitions_aug(st, a)
        items = list(dist.items())
        next_states = [s2 for s2, _ in items]
        next_probs  = np.array([p2 for _, p2 in items], dtype=float)
        next_probs  = next_probs / next_probs.sum()

        j = rng.choice(len(next_states), p=next_probs)
        st = next_states[j]

        base_traj.append(st[0])
        aug_traj.append(st)

    return base_traj, aug_traj
def print_policy_for_trajectory(
    mdp_aug: AugmentedMDPBaseline,
    policy_aug: PolicyAug,
    aug_traj: List[AugState],
) -> None:

    print("\nPolicy along visited augmented states:\n")

    for st in aug_traj:

        s = st[0]
        bits = st[1:]

        if mdp_aug.is_absorbing_aug(st):
            print(f"{s}  {bits}  -> GOAL")
            continue

        probs = policy_aug.get(st, {})

        if not probs:
            print(f"{s}  {bits}  -> no policy")
            continue

        best_a = max(probs, key=probs.get)

        print(
            f"{s}  {bits}  -> {best_a}   "
            f"(p={probs[best_a]:.3f})"
        )

def print_policy_for_visited_states(
    mdp_aug,
    policy_aug,
    aug_traj,
):
    arrow = {"U": "↑", "D": "↓", "L": "←", "R": "→"}

    state_to_arrow = {}

    for st in aug_traj:

        if mdp_aug.is_absorbing_aug(st):
            continue

        s = st[0]

        probs = policy_aug.get(st, {})
        if not probs:
            continue

        best_a = max(probs, key=probs.get)
        state_to_arrow[s] = arrow[best_a]

    for r in range(mdp_aug.base.N):
        row = ""

        for c in range(mdp_aug.base.N):
            s = (r, c)

            if s == mdp_aug.base.start:
                row += " S  "

            elif s in mdp_aug.base.goal:
                row += " G  "

            elif s in state_to_arrow:
                row += f" {state_to_arrow[s]}  "

            else:
                row += " ·  "

        print(row)        

def rollout_heatmap_aug(
    mdp_aug,
    policy_aug,
    n_rollouts: int = 50000,
    max_steps: int = 300,
    seed: int = 0,
    greedy: bool = False,
    save_path: str = "rollout_heatmap.png",
    show: bool = False,
):
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    N = mdp_aug.base.N

    # Counts how many rollouts visited each cell at least once
    visit_once_counts = np.zeros((N, N), dtype=float)

    reached_goal = 0
    total_steps = 0

    flag_hits = {spec.name: 0 for spec in mdp_aug.flags}
    until_success = {spec.name: 0 for spec in mdp_aug.until_specs}
    until_fail = {spec.name: 0 for spec in mdp_aug.until_specs}

    for k in range(n_rollouts):
        st = mdp_aug.start_aug
        visited_this_rollout = set()

        for t in range(max_steps + 1):
            s = st[0]
            visited_this_rollout.add(s)

            if mdp_aug.is_absorbing_aug(st):
                if mdp_aug.base.is_goal(s):
                    reached_goal += 1
                total_steps += t

                bits = st[1:]

                for spec in mdp_aug.flags:
                    idx = mdp_aug.flag_indices[spec.name]
                    if bits[idx] == 1:
                        flag_hits[spec.name] += 1

                for uspec in mdp_aug.until_specs:
                    i_succ = mdp_aug.until_success_idx[uspec.name]
                    i_fail = mdp_aug.until_fail_idx[uspec.name]

                    if bits[i_succ] == 1:
                        until_success[uspec.name] += 1
                    if bits[i_fail] == 1:
                        until_fail[uspec.name] += 1

                break

            probs = policy_aug.get(st, None)
            if probs is None or len(probs) == 0:
                total_steps += t
                break

            actions = list(probs.keys())
            p = np.array([probs[a] for a in actions], dtype=float)

            if p.sum() <= 1e-12:
                p = np.ones(len(actions), dtype=float) / len(actions)
            else:
                p = p / p.sum()

            if greedy:
                a = actions[int(np.argmax(p))]
            else:
                a = actions[int(rng.choice(len(actions), p=p))]

            dist = mdp_aug.transitions_aug(st, a)
            next_states = list(dist.keys())
            next_probs = np.array(list(dist.values()), dtype=float)
            next_probs = next_probs / next_probs.sum()

            st = next_states[int(rng.choice(len(next_states), p=next_probs))]

        # After one rollout finishes, add 1 to each cell visited at least once
        for r, c in visited_this_rollout:
            visit_once_counts[r, c] += 1.0

    visit_probs = visit_once_counts / n_rollouts

    stats = {
        "n_rollouts": n_rollouts,
        "empirical_p_goal": reached_goal / n_rollouts,
        "avg_steps": total_steps / n_rollouts,
        "empirical_flag_probs": {
            name: count / n_rollouts for name, count in flag_hits.items()
        },
        "empirical_until_success_probs": {
            name: count / n_rollouts for name, count in until_success.items()
        },
        "empirical_until_fail_probs": {
            name: count / n_rollouts for name, count in until_fail.items()
        },
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(visit_probs, origin="upper", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="P(cell visited at least once)")

    ax.set_title(f"Rollout visitation probability ({n_rollouts:,} runs)")
    ax.set_xlabel("column")
    ax.set_ylabel("row")

    sr, sc = mdp_aug.base.start
    ax.text(sc, sr, "S", ha="center", va="center", fontsize=12, fontweight="bold")

    for gr, gc in mdp_aug.base.goal:
        ax.text(gc, gr, "G", ha="center", va="center", fontsize=12, fontweight="bold")

    for spec in mdp_aug.flags:
        label = spec.name.replace("G", "")
        for rr, cc in spec.region:
            ax.text(cc, rr, label, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return visit_probs, stats      