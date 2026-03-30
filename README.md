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
conda create -n ekaterine python=3.10
conda activate ekaterine
