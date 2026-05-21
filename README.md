# 6G6Z0030 — HPC and Big Data Coursework

Challa Harikrishna Nagasai Charan
MMU ID: 23727842

This repo contains the code, scripts and supporting files for my 6G6Z0030 HPC and Big Data submission. 

## Folders

- `FinalReport/` — final PDF (same one submitted to Moodle) and the DOCX it came from.
- `HPC/OpenMP/` — OpenMP-optimised C driver, the autogen routines it links against, and the sweep script.
- `HPC/MPI/` — MPI version of the driver, the sweep script, and the Copilot prompt log.
- `HPC/Results/` — timing logs and the sweep CSVs produced by the two sweep scripts. `baseline_1thread.log` is the very first serial run; the actual table numbers in the report come from `openmp_sweep.csv` and `mpi_sweep.csv`.
- `BigData/Notebook/` — the Colab notebook used for the Spark analysis, exported with outputs preserved.
- `BigData/Results/` — any saved Spark outputs / intermediate files.
- `Graphs/` — the two figures used in the report (OpenMP time vs threads, and observed vs Amdahl speed-up).
- `Appendix/` — anything referenced from the report appendix that didn't sit naturally elsewhere.

## HPC code

The OpenMP driver in `HPC/OpenMP/fullStat_driver-OPT.c` was built from the supplied baseline with parallel reductions added on the vector stats and the power iteration loop, plus a private accumulator on the matvec to avoid false sharing on `z[]`. Compile with `-O0` as required by the brief — there's a one-liner in the report's Appendix A.2.

The MPI driver in `HPC/MPI/stats_mpi.c` was started from a Copilot-generated skeleton (prompts in `HPC/MPI/copilot_prompts.txt`) and then fixed by hand. The main fix was replacing an incorrect `MPI_Allgather` with `MPI_Allreduce(MPI_SUM)` on the partial `z[]` — explained in Q2c of the report.

Both `run_sweep.sh` (OpenMP) and `run_mpi_sweep.sh` (MPI) do three repeats per thread/process count and generate timing outputs inside the `results/` folders.
## Notebook

`BigData/Notebook/amazon_reviews_q2_2014_analysis.ipynb` is a Colab notebook. It loads the Amazon Electronics reviews dataset from Kaggle, filters to Q2 2014, runs the per-product comparison, and at the end runs a Mann-Whitney U test on the per-product review counts. Outputs are saved with the notebook so you can read it without re-running.

## Results and graphs

Anything quoted as a number in the report comes from either `HPC/Results/` (timings) or the notebook outputs (Big Data). Figures 1 and 2 in the report are also in `Graphs/` as PNGs.

## Submission

The PDF I uploaded to Moodle is `FinalReport/23727842_HPC_BigData_Final.pdf`. The repo URL is also listed in Appendix A.4 of the report and tutors have been invited as collaborators.
