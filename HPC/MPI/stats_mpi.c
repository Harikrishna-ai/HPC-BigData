#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// function prototypes
int get_num_data_points(FILE*);
int read_data(FILE*, int, double*, double*);

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    double *x = NULL;
    double *A = NULL;

    double startTotal = MPI_Wtime();

    // =========================
    // Rank 0 reads data
    // =========================
    if (rank == 0) {
        FILE *filePtr = fopen(argv[1], "r");

        if (filePtr == NULL) {
            printf("Cannot open file\n");
            n = -1;
        } else {
            int totalNum = get_num_data_points(filePtr);

            x = (double*) malloc(totalNum * sizeof(double));
            A = (double*) malloc((size_t)totalNum * totalNum * sizeof(double));

            double start_read = MPI_Wtime();
            n = read_data(filePtr, totalNum, x, A);

            printf("%d data points read [%f sec]\n", n, MPI_Wtime() - start_read);

            fclose(filePtr);
        }
    }

    // Broadcast n
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n < 0) {
        MPI_Finalize();
        return -1;
    }

    // Allocate on other ranks
    if (rank != 0) {
        x = (double*) malloc(n * sizeof(double));
        A = (double*) malloc((size_t)n * n * sizeof(double));
    }

    // Broadcast data
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(A, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // =========================
    // VECTOR STATS
    // =========================

    double local_sum = 0.0;
    for (int i = rank; i < n; i += size) {
        local_sum += x[i];
    }

    double sum = 0.0;
    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double mean = sum / (double)n;

    // variance
    local_sum = 0.0;
    for (int i = rank; i < n; i += size) {
        double v = x[i] - mean;
        local_sum += v * v;
    }

    double variance = 0.0;
    MPI_Allreduce(&local_sum, &variance, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    variance /= n;

    // min / max
    double local_min = fabs(x[rank]);
    double local_max = fabs(x[rank]);

    for (int i = rank; i < n; i += size) {
        double v = fabs(x[i]);
        if (v < local_min) local_min = v;
        if (v > local_max) local_max = v;
    }

    double minabs, maxabs;
    MPI_Allreduce(&local_min, &minabs, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &maxabs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // =========================
    // EIGENVALUE (POWER METHOD)
    // =========================

    double *approx = (double*) malloc(n * sizeof(double));
    double *z = (double*) malloc(n * sizeof(double));
    double *local_z = (double*) malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) approx[i] = 1.0;

    double tol = 0.01;
    double errmax;
    int iter = 0;
    double zmax;

    do {
        iter++;

        for (int i = 0; i < n; i++) local_z[i] = 0.0;

        for (int i = rank; i < n; i += size) {
            for (int j = 0; j < n; j++) {
                int pos = i + n*j;
                local_z[i] += A[pos] * approx[j];
            }
        }

        MPI_Allreduce(local_z, z, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double local_zmax = fabs(z[0]);
        for (int i = 1; i < n; i++) {
            if (fabs(z[i]) > local_zmax)
                local_zmax = fabs(z[i]);
        }

        MPI_Allreduce(&local_zmax, &zmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        for (int i = 0; i < n; i++) z[i] /= zmax;

        double local_err = 0.0;
        for (int i = 0; i < n; i++) {
            double e = fabs(fabs(z[i]) - fabs(approx[i]));
            if (e > local_err) local_err = e;
        }

        MPI_Allreduce(&local_err, &errmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        for (int i = 0; i < n; i++) approx[i] = z[i];

    } while (errmax > tol);

    // =========================
    // OUTPUT
    // =========================

    if (rank == 0) {
        printf("Stats:\n");
        printf("Min: %f Max: %f Mean: %f Std: %f\n",
               minabs, maxabs, mean, sqrt(variance));

        printf("Eigenvalue: %f (%d iterations)\n", zmax, iter);
        printf("Total time: %f\n", MPI_Wtime() - startTotal);
    }

    // cleanup
    free(x);
    free(A);
    free(approx);
    free(z);
    free(local_z);

    MPI_Finalize();
    return 0;
}
