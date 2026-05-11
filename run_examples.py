from __future__ import annotations

import argparse
import math
import random
import yaml
from typing import Any

from .src.examples import (
    simple_world,
    stochastic_world,
    wall_penalty_world,
    risky_corridor_world,
)
from .src.gridworld import (
    make_grid_world,
    print_cost_grid,
    make_4x4_pctl_world,  
    print_regions_grid
)
from .solvers.dp_solver import (
    value_iteration_shortest_path,
    greedy_policy_from_V,
    simulate_policy,
    simulate_policy_stochastic,
)
from .solvers.lp_solver import (
    solve_shortest_path_lp_gurobi,
    recover_policy_from_x_gurobi,
    print_policy_grid_gurobi,
)

from .solvers.pctl_solvers import (
    RegionFlagSpec,
    PCTLRegionConstraint,
    UntilSpec,
    UntilConstraint,
    AugmentedMDPBaseline,
    solve_lp_with_pctl_aug_baseline,
    recover_policy_from_x_aug,
    print_policy_grid_z0,
    simulate_policy_aug,
    print_policy_for_trajectory,
    print_policy_for_visited_states,
    rollout_heatmap_aug
)


WORLD_BUILDERS = {
    "simple": simple_world,
    "stochastic": stochastic_world,
    "wall_penalty": wall_penalty_world,
    "risky_corridor": risky_corridor_world,
    "4x4_pctl": make_4x4_pctl_world,
}


def is_bad_number(x: Any) -> bool:
    if x is None:
        return True
    try:
        return not math.isfinite(float(x))
    except Exception:
        return True


def build_world(name: str) -> Any:
    if name not in WORLD_BUILDERS:
        raise ValueError(
            f"Unknown world '{name}'. Choose from: {', '.join(WORLD_BUILDERS.keys())}"
        )
    return WORLD_BUILDERS[name]()


def build_mdp_from_yaml(path: str) -> tuple[Any, dict[str, Any]]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if "mdp" not in cfg:
        raise ValueError("YAML file must contain a top-level 'mdp' section.")

    mdp_cfg = cfg["mdp"]

    # Case 1: predefined world, e.g. world: "4x4_pctl"
    if "world" in mdp_cfg:
        mdp = build_world(mdp_cfg["world"])
        return mdp, cfg

    # Case 2: fully custom YAML world
    for key in ["N", "start", "goal"]:
        if key not in mdp_cfg:
            raise ValueError(f"YAML mdp section must contain '{key}'.")

    cell_costs = {
        (int(r), int(c)): float(cost)
        for (r, c, cost) in mdp_cfg.get("cell_costs", [])
    }

    rect_costs = [
        (int(r0), int(c0), int(r1), int(c1), float(cost))
        for (r0, c0, r1, c1, cost) in mdp_cfg.get("rect_costs", [])
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


def parse_region(spec: dict[str, Any], N: int) -> set[tuple[int, int]]:
    if "rect" in spec:
        r0, c0, r1, c1 = spec["rect"]
        return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}

    if "cells" in spec:
        return {tuple(cell) for cell in spec["cells"]}

    if "union" in spec:
        out = set()
        for part in spec["union"]:
            out |= parse_region(part, N)
        return out

    raise ValueError(f"Unknown region spec: {spec}")


def build_pctl_from_yaml(mdp: Any, pctl_cfg: dict[str, Any]) -> tuple[Any, float, list[Any], list[Any]]:
    regions = {}

    for f in pctl_cfg.get("flags", []):
        name = f["name"]
        regions[name] = parse_region(f["region"], mdp.N)

    flags = [
        RegionFlagSpec(name=name, region=regions[name])
        for name in regions
    ]

    until_specs = []
    for us in pctl_cfg.get("until_specs", []):
        until_specs.append(
            UntilSpec(
                name=us["name"],
                A_region=regions[us["A"]],
                B_region=regions[us["B"]],
            )
        )

    region_constraints = []
    for rc in pctl_cfg.get("region_constraints", []):
        region_constraints.append(
            PCTLRegionConstraint(
                kind=rc["type"],
                region_name=rc["region"],
                bound=float(rc["p"]),
            )
        )

    until_constraints = []
    for uc in pctl_cfg.get("until_constraints", []):
        until_constraints.append(
            UntilConstraint(
                kind=uc["type"],
                spec_name=uc["until"],
                bound=float(uc["p"]),
            )
        )

    mdp_aug = AugmentedMDPBaseline(
        base=mdp,
        flags=flags,
        until_specs=until_specs,
    )

    p_goal_min = float(pctl_cfg.get("p_goal_min", 1.0))

    return mdp_aug, p_goal_min, region_constraints, until_constraints, regions


def run_standard(mdp: Any, args: Any, rng: Any) -> None:
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

    J_lp, x_opt, solve_time = solve_shortest_path_lp_gurobi(
        mdp, verbose=args.verbose_lp
    )

    print("\n[LP] Objective value:")
    print(J_lp)
    print("[LP] Solve time:")
    print(solve_time)

    if J_lp is not None:
        diff = abs(V[mdp.start] - J_lp)
        print("\n[Check] |DP(start) - LP| =")
        print(diff)

    if x_opt is not None:
        pi_lp = recover_policy_from_x_gurobi(mdp, x_opt)

        print("\n[LP] Policy grid:")
        print_policy_grid_gurobi(mdp, pi_lp)

        


def run_pctl_lp(mdp: Any, cfg: dict[str, Any] | None, args: Any) -> None:
    if cfg is None or "pctl" not in cfg:
        raise ValueError(
            "PCTL mode requires a YAML config with a top-level 'pctl' section."
        )

    mdp_aug, p_goal_min, region_constraints, until_constraints, regions = build_pctl_from_yaml(
        mdp, cfg["pctl"]
    )

    J, p_goal, x_opt, region_probs, until_probs, t = solve_lp_with_pctl_aug_baseline(
        mdp_aug,
        p_goal_min=p_goal_min,
        region_constraints=region_constraints,
        until_constraints=until_constraints,
    )

    if x_opt is None or is_bad_number(J) or is_bad_number(p_goal) or len(x_opt) == 0:
        print("\n[PCTL LP] infeasible or failed.")
        return

    print("\n=== PCTL LP ===")
    print("Objective:", J)
    print("P(reach GOAL):", p_goal)

    if region_constraints:
        for name, val in region_probs.items():
            print(f"P(ever visit {name}): {val}")

    for name, val in until_probs.items():
        print(f"P({name}): {val}")

    print("Solve time:", t)

    pi_aug = recover_policy_from_x_aug(mdp_aug, x_opt)



    base_traj, aug_traj = simulate_policy_aug(
        mdp_aug,
        pi_aug,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    print("\nPolicy on visited states:")
    print_policy_for_visited_states(
        mdp_aug,
        pi_aug,
        aug_traj,
    )
    print_policy_for_trajectory(
        mdp_aug,
        pi_aug,
        aug_traj,
    )
    print_regions_grid(
        N=mdp.N,
        start=mdp.start,
        goal=mdp.goal,
        regions={
            name.replace("G", ""): region
            for name, region in regions.items()
        },
    )
    
    print("\n[PCTL LP] Example trajectory:")
    print(base_traj)


    if args.rollout_heatmap:
        import os

        os.makedirs(os.path.dirname(args.heatmap_path) or ".", exist_ok=True)

        print("\n[ROLLOUT HEATMAP] Starting rollouts...", flush=True)

        visits, stats = rollout_heatmap_aug(
            mdp_aug=mdp_aug,
            policy_aug=pi_aug,
            n_rollouts=args.n_rollouts,
            max_steps=args.max_rollout_steps,
            seed=args.seed,
            greedy=False,
            save_path=args.heatmap_path,
            show=False,
        )

        print("\n[ROLLOUT HEATMAP]", flush=True)
        print(f"Saved to: {args.heatmap_path}", flush=True)
        print(f"Empirical P(reach GOAL): {stats['empirical_p_goal']:.4f}", flush=True)
        print(f"Average steps: {stats['avg_steps']:.2f}", flush=True)

        for name, val in stats["empirical_flag_probs"].items():
            print(f"Empirical P(ever visit {name}): {val:.4f}", flush=True)

        for name, val in stats["empirical_until_success_probs"].items():
            print(f"Empirical P({name} success): {val:.4f}", flush=True)

        for name, val in stats["empirical_until_fail_probs"].items():
            print(f"Empirical P({name} fail): {val:.4f}", flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SSP/PCTL examples for Ekaterine.")

    parser.add_argument(
        "--world",
        type=str,
        default="stochastic",
        choices=WORLD_BUILDERS.keys(),
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--verbose_lp", action="store_true")

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["standard", "pctl"],
        help="Optional override. If omitted, uses run.solver from YAML if present.",
    )
    parser.add_argument(
        "--rollout-heatmap",
        action="store_true",
        help="Run Monte Carlo rollouts and save a visitation heatmap.",
    )

    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=50000,
        help="Number of rollouts for heatmap.",
    )

    parser.add_argument(
        "--heatmap-path",
        type=str,
        default="rollout_heatmap.png",
        help="Path to save rollout heatmap.",
    )

    parser.add_argument(
        "--max-rollout-steps",
        type=int,
        default=300,
        help="Maximum steps per rollout.",
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)

    if args.config is not None:
        mdp, cfg = build_mdp_from_yaml(args.config)
        world_name = f"yaml:{args.config}"
    else:
        mdp = build_world(args.world)
        cfg = None
        world_name = args.world

    run_cfg = cfg.get("run", {}) if cfg is not None else {}
    solver = run_cfg.get("solver", "standard").lower()

    if args.mode is not None:
        solver = "pctl_lp" if args.mode == "pctl" else "standard"

    print("=" * 60)
    print(f"World: {world_name}")
    print(f"Grid size: {mdp.N}x{mdp.N}")
    print(f"Start: {mdp.start}")
    print(f"Goal: {mdp.goal}")
    print(f"Default slip_prob: {mdp.slip_prob}")
    print(f"Solver/mode: {solver}")
    print("=" * 60)

    print("\nCost grid:")
    print_cost_grid(mdp)

    if solver in ("standard", "dp", "lp", "both"):
        run_standard(mdp, args, rng)

    elif solver == "pctl_lp":
        run_pctl_lp(mdp, cfg, args)

    else:
        raise ValueError(
            "Unknown solver/mode. Use standard | dp | lp | both | pctl_lp."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()