#!/bin/bash
mkdir -p results
echo "threads,repeat,total_time,eigen_time,vector_time,read_time" > results/openmp_sweep.csv

for T in 1 2 4 6 8 12; do
  for R in 1 2 3; do
    export OMP_NUM_THREADS=$T
    echo "=== Threads=$T Repeat=$R ==="
    OUT=$(./stats_opt 25K.txt)
    echo "$OUT" | tee results/opt_t${T}_r${R}.log
    TOTAL=$(echo "$OUT"  | awk '/Total time:/        {print $3}')
    EIGEN=$(echo "$OUT"  | awk '/Time eigenvalue:/   {print $3}')
    VEC=$(echo "$OUT"    | awk '/Time vector stats:/ {print $4}')
    READ=$(echo "$OUT"   | awk '/Time data read:/    {print $4}')
    echo "$T,$R,$TOTAL,$EIGEN,$VEC,$READ" >> results/openmp_sweep.csv
  done
done
echo "DONE - results in results/openmp_sweep.csv"
cat results/openmp_sweep.csv
