from __future__ import annotations

from typing import Dict, Optional, List
import random

from .gridworld import GridWorld, State, Action


# =========================
# DP helpers (ground truth)
# =========================

def q_value(mdp: GridWorld, V: Dict[State, float], s: State, a: Action, gamma: float = 1.0) -> float:
    """
    Bellman Q-value under the true stochastic transition kernel:

        Q(s,a) = E[ cost(s,a,s') + gamma * V(s') ]
    """
    exp_next_V = sum(p * V[s2] for s2, p in mdp.transitions(s, a).items())
    return mdp.cost(s, a) + gamma * exp_next_V


def value_iteration_shortest_path(
    mdp: GridWorld,
    gamma: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 10_000,
    init_value: float = 1e6,
) -> Dict[State, float]:
    """
    Value iteration for stochastic shortest path / discounted control.

    For goal states:
        V(goal) = 0

    For non-goal states:
        V(s) = min_a [ mdp.cost(s,a) + gamma * sum_{s'} T(s,a,s')V(s') ]
    """
    V: Dict[State, float] = {
        s: 0.0 if mdp.is_goal(s) else float(init_value)
        for s in mdp.states
    }

    for _ in range(max_iter):
        delta = 0.0
        V_new = V.copy()

        for s in mdp.states:
            if mdp.is_goal(s):
                continue

            actions = mdp.actions_from(s)
            if not actions:
                continue

            best_q = min(q_value(mdp, V, s, a, gamma=gamma) for a in actions)
            V_new[s] = best_q
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new

        if delta < tol:
            break

    return V


def greedy_policy_from_V(
    mdp: GridWorld,
    V: Dict[State, float],
    gamma: float = 1.0,
    tol: float = 1e-12,
) -> Dict[State, Optional[Action]]:
    """
    Deterministic greedy policy from a value function.
    """
    pi: Dict[State, Optional[Action]] = {}

    for s in mdp.states:
        if mdp.is_goal(s):
            pi[s] = None
            continue

        best_a = None
        best_q = float("inf")

        for a in mdp.actions_from(s):
            q = q_value(mdp, V, s, a, gamma=gamma)
            if q < best_q - tol:
                best_q = q
                best_a = a

        pi[s] = best_a

    return pi


def stochastic_greedy_policy_from_V(
    mdp: GridWorld,
    V: Dict[State, float],
    gamma: float = 1.0,
    tol: float = 1e-10,
) -> Dict[State, Dict[Action, float]]:
    """
    Return a tie-aware stochastic greedy policy:
    if multiple actions achieve the same best Q up to tol,
    split probability uniformly among them.
    """
    pi: Dict[State, Dict[Action, float]] = {}

    for s in mdp.states:
        if mdp.is_goal(s):
            pi[s] = {}
            continue

        qs = {}
        for a in mdp.actions_from(s):
            qs[a] = q_value(mdp, V, s, a, gamma=gamma)

        best_q = min(qs.values())
        best_actions = [a for a, q in qs.items() if abs(q - best_q) <= tol]

        p = 1.0 / len(best_actions)
        pi[s] = {a: (p if a in best_actions else 0.0) for a in mdp.actions_from(s)}

    return pi


def simulate_policy(
    mdp: GridWorld,
    pi: Dict[State, Optional[Action]],
    max_steps: int = 500,
) -> List[State]:
    """
    Deterministic rollout using mdp.move() and a deterministic policy.
    """
    s = mdp.start
    traj = [s]

    for _ in range(max_steps):
        a = pi.get(s, None)
        if a is None:
            break

        s = mdp.move(s, a)
        traj.append(s)

        if mdp.is_goal(s):
            break

    return traj


def simulate_policy_stochastic(
    mdp: GridWorld,
    pi: Dict[State, Optional[Action]],
    max_steps: int = 500,
    rng: Optional[random.Random] = None,
) -> List[State]:
    """
    Stochastic rollout under the actual transition kernel mdp.transitions().
    """
    if rng is None:
        rng = random.Random()

    s = mdp.start
    traj = [s]

    for _ in range(max_steps):
        a = pi.get(s, None)
        if a is None:
            break

        dist = mdp.transitions(s, a)
        next_states = list(dist.keys())
        probs = list(dist.values())

        s = rng.choices(next_states, weights=probs, k=1)[0]
        traj.append(s)

        if mdp.is_goal(s):
            break

    return traj


def expected_cost_of_policy(
    mdp: GridWorld,
    pi: Dict[State, Optional[Action]],
    gamma: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 10_000,
) -> Dict[State, float]:
    """
    Policy evaluation for a fixed deterministic policy.

    Solves:
        V^pi(s) = cost(s,pi(s)) + gamma * sum_{s'} T(s,pi(s),s') V^pi(s')
    with V^pi(goal) = 0
    by simple iterative evaluation.
    """
    V: Dict[State, float] = {
        s: 0.0 if mdp.is_goal(s) else 0.0
        for s in mdp.states
    }

    for _ in range(max_iter):
        delta = 0.0
        V_new = V.copy()

        for s in mdp.states:
            if mdp.is_goal(s):
                continue

            a = pi.get(s, None)
            if a is None:
                continue

            V_new[s] = q_value(mdp, V, s, a, gamma=gamma)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        if delta < tol:
            break

    return V