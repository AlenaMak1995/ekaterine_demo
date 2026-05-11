#!/bin/bash
set -e

cd /nfs/hpc/share/makarova/ekaterine

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p ekaterine_demo/results

python -m ekaterine_demo.run_examples \
  --config ekaterine_demo/configs/10x10_pctl_test.yaml \
  --rollout-heatmap \
  --n-rollouts 50000 \
  --heatmap-path ekaterine_demo/results/10x10_slip02_heatmap.png