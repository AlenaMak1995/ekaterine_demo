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

####  What the script does

For a given MDP:

Solves using DP (value iteration)
Solves using LP (Gurobi)
Compares results:
DP value at start
LP objective
Prints:
cost grid
sample trajectories
recovered LP policy
