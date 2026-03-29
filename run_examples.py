from __future__ import annotations

import argparse
import random
import yaml

from .examples import (
    simple_world,
    stochastic_world,
    wall_penalty_world,
    risky_corridor_world,
)
from .gridworld import make_grid_world, print_cost_grid
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


def build_mdp_from_yaml(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if "mdp" not in cfg:
        raise ValueError("YAML file must contain a top-level 'mdp' section.")

    mdp_cfg = cfg["mdp"]

    if "N" not in mdp_cfg:
        raise ValueError("YAML mdp section must contain 'N'.")
    if "start" not in mdp_cfg:
        raise ValueError("YAML mdp section must contain 'start'.")
    if "goal" not in mdp_cfg:
        raise ValueError("YAML mdp section must contain 'goal'.")

    # parse cell_costs in format:
    # cell_costs:
    #   - [r, c, cost]
    #   - [r, c, cost]
    cell_costs_list = mdp_cfg.get("cell_costs", [])
    cell_costs = {
        (int(r), int(c)): float(cost)
        for (r, c, cost) in cell_costs_list
    }

    # parse rect_costs in format:
    # rect_costs:
    #   - [r0, c0, r1, c1, cost]
    rect_costs_list = mdp_cfg.get("rect_costs", [])
    rect_costs = [
        (int(r0), int(c0), int(r1), int(c1), float(cost))
        for (r0, c0, r1, c1, cost) in rect_costs_list
    ]

    mdp = make_grid_world(
        N=int(mdp_cfg["N"]),
        start=tuple(mdp_cfg["start"]),
        goal={tuple(g) for g in mdp_cfg["goal"]},
        default_cost=float(mdp_cfg.get("default_cost", 1.0)),
        cell_costs=cell_costs,
        rect_costs=rect_costs,
        slip_prob=float(mdp_cfg.get("slip_prob", 0.0)),
    )

    return mdp, cfg


def main():
    parser = argparse.ArgumentParser(description="Run SSP examples for Ekaterine.")

    parser.add_argument(
        "--world",
        type=str,
        default="stochastic",
        choices=WORLD_BUILDERS.keys(),
        help="Which predefined example world to run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to YAML config. If given, overrides --world.",
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
    if args.config is not None:
        mdp, cfg = build_mdp_from_yaml(args.config)
        world_name = f"yaml:{args.config}"
    else:
        mdp = build_world(args.world)
        cfg = None
        world_name = args.world

    print("=" * 60)
    print(f"World: {world_name}")
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