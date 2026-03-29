from __future__ import annotations

import argparse
import random

from .examples import (
    simple_world,
    stochastic_world,
    wall_penalty_world,
    risky_corridor_world,
)
from .gridworld import print_cost_grid
from .dp_solver import (
    value_iteration_shortest_path,
    greedy_policy_from_V,
    simulate_policy,
    simulate_policy_stochastic,
)
from .lp_solver import (
    solve_shortest_path_lp_gurobi,
    recover_policy_from_x_gurobi,
    print_policy_grid_gurobi,
)


WORLD_BUILDERS = {
    "simple": simple_world,
    "stochastic": stochastic_world,
    "wall_penalty": wall_penalty_world,
    "risky_corridor": risky_corridor_world,
}


def build_world(name: str):
    if name not in WORLD_BUILDERS:
        raise ValueError(
            f"Unknown world '{name}'. "
            f"Choose from: {', '.join(WORLD_BUILDERS.keys())}"
        )
    return WORLD_BUILDERS[name]()


def main():
    parser = argparse.ArgumentParser(description="Run SSP examples for Ekaterine.")
    parser.add_argument(
        "--world",
        type=str,
        default="stochastic",
        choices=WORLD_BUILDERS.keys(),
        help="Which example world to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for stochastic rollout.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Max rollout length.",
    )
    parser.add_argument(
        "--verbose_lp",
        action="store_true",
        help="Show Gurobi output.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1) Build world
    mdp = build_world(args.world)

    print("=" * 60)
    print(f"World: {args.world}")
    print(f"Grid size: {mdp.N}x{mdp.N}")
    print(f"Start: {mdp.start}")
    print(f"Goal: {mdp.goal}")
    print(f"Default slip_prob: {mdp.slip_prob}")
    print("=" * 60)

    print("\nCost grid:")
    print_cost_grid(mdp)

    # 2) DP benchmark
    V = value_iteration_shortest_path(mdp)
    pi_dp = greedy_policy_from_V(mdp, V)

    print("\n[DP] Optimal value at start:")
    print(V[mdp.start])

    print("\n[DP] Greedy policy rollout using deterministic move():")
    traj_det = simulate_policy(mdp, pi_dp, max_steps=args.max_steps)
    print(traj_det)

    print("\n[DP] One stochastic rollout using true transitions():")
    traj_stoch = simulate_policy_stochastic(
        mdp, pi_dp, max_steps=args.max_steps, rng=rng
    )
    print(traj_stoch)

    # 3) LP benchmark
    J_lp, x_opt, solve_time = solve_shortest_path_lp_gurobi(
        mdp, verbose=args.verbose_lp
    )

    print("\n[LP] Objective value:")
    print(J_lp)
    print("[LP] Solve time:")
    print(solve_time)

    # 4) Compare DP vs LP
    if J_lp is not None:
        diff = abs(V[mdp.start] - J_lp)
        print("\n[Check] |DP(start) - LP| =")
        print(diff)

    # 5) Recover LP policy
    if x_opt is not None:
        pi_lp = recover_policy_from_x_gurobi(mdp, x_opt)

        print("\n[LP] Policy grid:")
        print_policy_grid_gurobi(mdp, pi_lp)

    print("\nDone.")


if __name__ == "__main__":
    main()