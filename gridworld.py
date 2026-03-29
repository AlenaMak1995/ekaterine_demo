from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple

Action = str
State = Tuple[int, int]
Region = Set[State]
SlipRule = Callable[[State, Action], float]

ACTIONS: List[Action] = ["U", "D", "L", "R"]

DELTA = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


@dataclass
class GridWorld:
    """
    Simple grid MDP.

    Attributes
    ----------
    N : int
        Grid size (N x N).
    start : State
        Initial cell.
    goal : Region
        Set of goal cells. Goal states are absorbing.
    cost_cell : callable
        Function cost_cell(r, c) -> float giving the cost of entering cell (r, c).
    slip_prob : float
        Default slip probability.
    slip_rule : callable | None
        Optional state/action-dependent slip rule. If provided, overrides slip_prob.
    """
    N: int
    start: State
    goal: Region
    cost_cell: Callable[[int, int], float]
    slip_prob: float = 0.0
    slip_rule: SlipRule | None = None

    def __post_init__(self) -> None:
        self.states: List[State] = [
            (r, c) for r in range(self.N) for c in range(self.N)
        ]

    def clamp(self, r: int, c: int) -> State:
        return max(0, min(self.N - 1, r)), max(0, min(self.N - 1, c))

    def move(self, s: State, a: Action) -> State:
        dr, dc = DELTA[a]
        r, c = s
        return self.clamp(r + dr, c + dc)

    def is_goal(self, s: State) -> bool:
        return s in self.goal

    def actions_from(self, s: State) -> List[Action]:
        if self.is_goal(s):
            return []
        return ACTIONS

    def slip(self, s: State, a: Action) -> float:
        if self.slip_rule is not None:
            return float(self.slip_rule(s, a))
        return float(self.slip_prob)

    def transitions(self, s: State, a: Action) -> Dict[State, float]:
        """
        Transition kernel P(s' | s, a).

        With slip = eps:
        - intended action gets probability 1 - eps
        - each of the other 3 actions gets eps / 3
        """
        if self.is_goal(s):
            return {s: 1.0}

        next_main = self.move(s, a)
        eps = self.slip(s, a)

        if eps <= 0.0:
            return {next_main: 1.0}

        probs: Dict[State, float] = {}
        probs[next_main] = probs.get(next_main, 0.0) + (1.0 - eps)

        for other_a in ACTIONS:
            if other_a == a:
                continue
            s2 = self.move(s, other_a)
            probs[s2] = probs.get(s2, 0.0) + eps / 3.0

        return probs

    def cost(self, s: State, a: Action) -> float:
        """
        Expected one-step cost of taking action a in state s.
        Cost is based on the realized next cell.
        """
        exp_cost = 0.0
        for s2, p in self.transitions(s, a).items():
            r2, c2 = s2
            exp_cost += p * self.cost_cell(r2, c2)
        return exp_cost


def make_grid_world(
    N: int,
    start: State,
    goal: Region,
    default_cost: float = 1.0,
    cell_costs: Dict[State, float] | None = None,
    rect_costs: List[Tuple[int, int, int, int, float]] | None = None,
    slip_prob: float = 0.0,
    slip_rule: SlipRule | None = None,
) -> GridWorld:
    """
    Build a gridworld with optional per-cell and rectangular cost overrides.

    Parameters
    ----------
    N : int
        Grid size.
    start : State
        Start cell.
    goal : set[State]
        Goal region.
    default_cost : float
        Default cell-entry cost.
    cell_costs : dict
        Explicit overrides {(r, c): cost}.
    rect_costs : list
        Rectangle overrides [(r0, c0, r1, c1, cost)] inclusive.
    slip_prob : float
        Global slip probability.
    slip_rule : callable | None
        Optional state/action-dependent slip rule.
    """
    cell_costs = cell_costs or {}
    rect_costs = rect_costs or []

    def cost_cell(r: int, c: int) -> float:
        if (r, c) in cell_costs:
            return float(cell_costs[(r, c)])

        for r0, c0, r1, c1, cost in rect_costs:
            if r0 <= r <= r1 and c0 <= c <= c1:
                return float(cost)

        return float(default_cost)

    return GridWorld(
        N=N,
        start=start,
        goal=goal,
        cost_cell=cost_cell,
        slip_prob=slip_prob,
        slip_rule=slip_rule,
    )


def print_cost_grid(mdp: GridWorld, digits: int = 1) -> None:
    """Print the cell-cost map."""
    for r in range(mdp.N):
        row = [f"{mdp.cost_cell(r, c):.{digits}f}" for c in range(mdp.N)]
        print("\t".join(row))
    print()


def make_example_world() -> GridWorld:
    """
    Tiny default example world for quick testing.
    """
    return make_grid_world(
        N=5,
        start=(4, 0),
        goal={(0, 4)},
        default_cost=1.0,
        cell_costs={
            (1, 1): 8.0,
            (1, 2): 8.0,
            (2, 2): 5.0,
            (3, 3): 10.0,
        },
        slip_prob=0.0,
    )