#!/bin/bash
mkdir -p results
echo "procs,repeat,total_time" > results/mpi_sweep.csv

for P in 1 2 4 6; do
  for R in 1 2 3; do
    echo "=== Procs=$P Repeat=$R ==="
    OUT=$(mpirun -np $P ./stats_mpi 25K.txt 2>&1)
    echo "$OUT" | tee results/mpi_p${P}_r${R}.log
    TOTAL=$(echo "$OUT" | awk '/Total time:/ {print $3}')
    echo "$P,$R,$TOTAL" >> results/mpi_sweep.csv
  done
done

echo "DONE"
cat results/mpi_sweep.csv
