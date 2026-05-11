# Ekaterine MDP Playground


This repository contains a minimal setup for solving a stochastic shortest path (SSP) problem in a gridworld.

It includes:
- a simple grid MDP with stochastic transitions (slip)
- a dynamic programming (DP) solver (ground truth)
- a linear programming (LP) solver using Gurobi
- a few example environments
- optional YAML configs for easy customization

---

## 📦 Installation

### 1. Create environment (recommended)

```bash
conda create -n <<environment name>> python=3.10
conda activate <<environment name>>
```


### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Gurobi 

This project uses Gurobi for the LP solver.

Install: pip install gurobipy
You need a valid license (academic license is free)

Test:

```bash
python -c "import gurobipy as gp; gp.Model()"
```

### 4. HPC note 

If running on a cluster, limit BLAS threads:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

### 5. Running examples

Run from the parent directory of ekaterine_demo:

```bash
python -m ekaterine_demo.run_examples --world stochastic
```

####  Available example worlds

```bash
--world simple
--world stochastic
--world wall_penalty
--world risky_corridor
```

### 6. Running with YAML config

You can also define your own environment via YAML:

```bash
python -m ekaterine_demo.run_examples --config ekaterine_demo/configs/stochastic.yaml
```
####  Example YAML

```bash
mdp:
  N: 5
  start: [4, 0]
  goal:
    - [0, 4]

  default_cost: 1.0
  slip_prob: 0.2
```

Optional:

```bash
cell_costs:
  - [2, 2, 20.0]

rect_costs:
  - [3, 0, 4, 4, 5.0]
```

#### What the script does

For a given MDP:

- **DP (value iteration)** → ground truth solution  
- **LP (Gurobi)** → occupancy-measure formulation  
- **Comparison**:
  - DP value at start
  - LP objective  
- **Outputs**:
  - cost grid
  - sample trajectories
  - recovered LP policy

---

#### Expected behavior

- DP and LP values should match closely:

```text
|DP(start) - LP| ≈ 1e-6
```
With slip → trajectories become stochastic

With high-cost regions → policy avoids risky areas

# UPDATE from 05/11/2026

## Running PCTL Examples

This project supports solving stochastic gridworld shortest-path problems with PCTL-style constraints using an augmented MDP and an LP formulation.

To run the 10x10 PCTL example:

```bash
python -m ekaterine_demo.run_examples \
  --config configs/10x10_pctl_test.yaml
```

The script prints:

- the LP objective value,
- the probability of reaching the goal,
- the probability of visiting each flagged region,
- the probability of satisfying each until formula,
- an example simulated trajectory,
- the policy on visited states.

```bash
Example output:

P(reach GOAL): 0.99999995
P(ever visit G2): 1.0000
P(ever visit G3): 0.2035
P(ever visit G4): 0.1462
P(G2U_G3): 0.2000
```

## Rollout Heatmap Diagnostics

To validate the LP policy by Monte Carlo simulation, run:

```bash
./scripts/run_10x10_heatmap.sh
```

This script:

1. solves the PCTL LP,
2. recovers the stochastic policy,
3. simulates many rollouts under that policy,
4. saves a heatmap of empirical state visitation probabilities.

By default, the heatmap is saved to:

```bash
results/10x10_slip02_heatmap.png
```

The heatmap colorbar shows:

```bash

P(cell is visited at least once during a rollout)
```

Values are between 0 and 1. For example, a value of 0.20 means that approximately 20% of rollouts visited that cell at least once.

The rollout output also prints empirical probabilities such as:

```bash
Empirical P(reach GOAL)
Empirical P(ever visit G2)
Empirical P(ever visit G3)
Empirical P(ever visit G4)
Empirical P(G2U_G3 success)
Empirical P(G2U_G3 fail)
```

These empirical probabilities can be compared with the LP probabilities printed by the solver.

Example comparison:

```bash
LP:
P(ever visit G3): 0.2035
P(ever visit G4): 0.1462
P(G2U_G3):        0.2000

Rollouts:
Empirical P(ever visit G3):       0.2051
Empirical P(ever visit G4):       0.1448
Empirical P(G2U_G3 success):      0.2016
Empirical P(G2U_G3 fail):         0.7984
```
## Running Rollouts Manually

The heatmap command can also be run manually:

```bash
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
python -m ekaterine_demo.run_examples \
  --config configs/10x10_pctl_test.yaml \
  --rollout-heatmap \
  --n-rollouts 50000 \
  --heatmap-path results/10x10_slip02_heatmap.png
```

For a faster test, use fewer rollouts:

```bash
python -m ekaterine_demo.run_examples \
  --config configs/10x10_pctl_test.yaml \
  --rollout-heatmap \
  --n-rollouts 1000 \
  --heatmap-path results/test_heatmap.png
```

