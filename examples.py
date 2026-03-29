from .gridworld import make_grid_world


def simple_world():
    return make_grid_world(
        N=5,
        start=(4, 0),
        goal={(0, 4)},
        default_cost=1.0,
        slip_prob=0.0,
    )


def stochastic_world():
    return make_grid_world(
        N=5,
        start=(4, 0),
        goal={(0, 4)},
        default_cost=1.0,
        slip_prob=0.2,
    )


def wall_penalty_world():
    return make_grid_world(
        N=5,
        start=(4, 0),
        goal={(0, 4)},
        default_cost=1.0,
        rect_costs=[
            (3, 0, 4, 4, 5.0),  # bottom rows = more expensive
        ],
        slip_prob=0.1,
    )


def risky_corridor_world():
    return make_grid_world(
        N=5,
        start=(4, 0),
        goal={(0, 4)},
        default_cost=1.0,
        cell_costs={
            (2, 2): 20.0,
            (2, 3): 20.0,
        },
        slip_prob=0.2,
    )