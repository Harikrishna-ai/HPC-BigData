#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int get_num_data_points(FILE*);
int read_data(FILE*, int, double*, double*);

int main(int argc, char** argv) {
  int n;
  double *x, *A;
  double *squaredDiffs;
  double startTotalCode = omp_get_wtime();

  FILE* filePtr;
  char *filename = argv[1];
  filePtr = fopen(filename, "r");
  if (filePtr == NULL) {
    printf("Cannot open file %s\n", filename);
    return -1;
  }

  int totalNum = get_num_data_points(filePtr);
  printf("There are allegedly %d data points to read\n", totalNum);

  x = (double *) malloc(totalNum * sizeof(double));
  A = (double *) malloc((size_t)totalNum * totalNum * sizeof(double));

  if (x == NULL || A == NULL) {
    printf("Memory allocation failed\n");
    return -1;
  }

  n = read_data(filePtr, totalNum, x, A);
  double afterRead = omp_get_wtime();

  squaredDiffs = (double *) malloc(n * sizeof(double));
  double *z = (double *) malloc(n * sizeof(double));
  double *err = (double *) malloc(n * sizeof(double));
  double *approx = (double *) malloc(n * sizeof(double));

  double sum = 0.0, mean;

  // Mean
  #pragma omp parallel for reduction(+:sum)
  for (int i=0; i<n; i++) sum += x[i];
  mean = sum / n;

  // Variance
  #pragma omp parallel for
  for (int i=0; i<n; i++) {
    double v = x[i] - mean;
    squaredDiffs[i] = v * v;
  }

  sum = 0.0;
  #pragma omp parallel for reduction(+:sum)
  for (int i=0; i<n; i++) sum += squaredDiffs[i];
  double variance = sum / n;

  double afterVec_start = omp_get_wtime();
  // Min / Max
  double minabs = fabs(x[0]);
  double maxabs = fabs(x[0]);

  #pragma omp parallel for reduction(min:minabs)
  for (int i=0; i<n; i++) {
    double v = fabs(x[i]);
    if (v < minabs) minabs = v;
  }

  #pragma omp parallel for reduction(max:maxabs)
  for (int i=0; i<n; i++) {
    double v = fabs(x[i]);
    if (v > maxabs) maxabs = v;
  }

  double afterMinMax = omp_get_wtime();
  // Eigenvalue
  for (int i=0; i<n; i++) approx[i] = 1.0;

  double tol = 0.01;
  int iterCount = 0;
  double zmax, errmax;

  do {
    iterCount++;

    #pragma omp parallel for
    for (int i=0; i<n; i++) {
      double zi = 0.0;
      for (int j=0; j<n; j++) {
        zi += A[i + (size_t)n*j] * approx[j];
      }
      z[i] = zi;
    }

    zmax = fabs(z[0]);
    #pragma omp parallel for reduction(max:zmax)
    for (int i=0; i<n; i++) {
      double v = fabs(z[i]);
      if (v > zmax) zmax = v;
    }

    #pragma omp parallel for
    for (int i=0; i<n; i++) z[i] /= zmax;

    #pragma omp parallel for
    for (int i=0; i<n; i++)
      err[i] = fabs(fabs(z[i]) - fabs(approx[i]));

    errmax = err[0];
    #pragma omp parallel for reduction(max:errmax)
    for (int i=1; i<n; i++)
      if (err[i] > errmax) errmax = err[i];

    #pragma omp parallel for
    for (int i=0; i<n; i++) approx[i] = z[i];

  } while (errmax > tol);
  double afterEigen = omp_get_wtime();

  printf("Stats:\n");
  printf("Min: %f Max: %f Mean: %f Std: %f\n",
         minabs, maxabs, mean, sqrt(variance));
  printf("Eigenvalue: %f (%d iterations)\n", zmax, iterCount);

  printf("Time data read:    %f\n", afterRead - startTotalCode);
  printf("Time vector stats: %f\n", afterMinMax - afterRead);
  printf("Time eigenvalue:   %f\n", afterEigen - afterMinMax);
  printf("Total time:        %f\n", omp_get_wtime() - startTotalCode);

  return 0;
}
