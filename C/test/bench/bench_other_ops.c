// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "tensor.h"

#define NUM_RUNS 10

void run_bench(int32_t size, int num_threads) {
    int32_t shape[] = {size};
    tensor *a = allocTensor(1, shape);
    tensor *b = allocTensor(1, shape);

    for (int32_t i = 0; i < size; i++) {
        a->buffer->data[i] = (float)(i % 10 - 5); // contiene anche valori negativi per relu
        b->buffer->data[i] = (float)(i % 5);
    }

    omp_set_num_threads(num_threads);

    // Warm-up
    tensor *w1 = relu(a);
    tensorDeref(w1);

    // Benchmark RELU (R)
    double relu_time = 0.0;
    for (int run = 0; run < NUM_RUNS; run++) {
        double start = omp_get_wtime();
        tensor *res = relu(a);
        double end = omp_get_wtime();
        relu_time += (end - start);
        tensorDeref(res);
    }

    // Benchmark MIN (m)
    double min_time = 0.0;
    for (int run = 0; run < NUM_RUNS; run++) {
        double start = omp_get_wtime();
        tensor *res = minTensor(a, b);
        double end = omp_get_wtime();
        min_time += (end - start);
        tensorDeref(res);
    }

    // Benchmark MAX (M)
    double max_time = 0.0;
    for (int run = 0; run < NUM_RUNS; run++) {
        double start = omp_get_wtime();
        tensor *res = maxTensor(a, b);
        double end = omp_get_wtime();
        max_time += (end - start);
        tensorDeref(res);
    }

    // Benchmark FILL (f)
    // Creiamo shape 1D di size
    int32_t sh_shape[] = {1};
    tensor *shape_t = allocTensor(1, sh_shape);
    shape_t->buffer->data[0] = (float)size;

    int32_t val_shape[] = {5};
    tensor *val_t = allocTensor(1, val_shape);
    for (int i = 0; i < 5; i++) val_t->buffer->data[i] = (float)i;

    double fill_time = 0.0;
    for (int run = 0; run < NUM_RUNS; run++) {
        double start = omp_get_wtime();
        tensor *res = fill(shape_t, val_t);
        double end = omp_get_wtime();
        fill_time += (end - start);
        tensorDeref(res);
    }

    // Benchmark RESHAPE (r) e RAVEL (_)
    // poichè queste operazioni non copiano memoria ma cambiano solo metadati, sono istantanee.
    // Le benchmarkiamo comunque per completezza
    double reshape_time = 0.0;
    for (int run = 0; run < NUM_RUNS; run++) {
        double start = omp_get_wtime();
        tensor *res = reshape(a, shape_t);
        double end = omp_get_wtime();
        reshape_time += (end - start);
        tensorDeref(res);
    }

    double relu_ms = (relu_time / NUM_RUNS) * 1000.0;
    double min_ms = (min_time / NUM_RUNS) * 1000.0;
    double max_ms = (max_time / NUM_RUNS) * 1000.0;
    double fill_ms = (fill_time / NUM_RUNS) * 1000.0;
    double reshape_ms = (reshape_time / NUM_RUNS) * 1000.0;

    printf("| %10d | %10d | %9.3f ms | %9.3f ms | %9.3f ms | %9.3f ms | %9.3f ms |\n",
           size, num_threads, relu_ms, min_ms, max_ms, fill_ms, reshape_ms);

    tensorDeref(a);
    tensorDeref(b);
    tensorDeref(shape_t);
    tensorDeref(val_t);
}

int main() {
    printf("=== BENCHMARK DI SCALABILITÀ OPERAZIONI VARIE (RELU, MIN, MAX, FILL, RESHAPE) ===\n");
    printf("Numero massimo di thread disponibili: %d\n\n", omp_get_max_threads());

    int32_t sizes[] = {10000, 1000000, 10000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    int thread_configs[] = {1, 2, 4, 8};
    int num_configs = sizeof(thread_configs) / sizeof(thread_configs[0]);
    int max_threads = omp_get_max_threads();

    for (int s = 0; s < num_sizes; s++) {
        int32_t size = sizes[s];
        printf("--------------------------------------------------------------------------------------------------\n");
        printf("Benchmark con Tensore di dimensione: %d elementi (%.2f MB per operando)\n", size, (double)size * sizeof(float) / 1024.0 / 1024.0);
        printf("--------------------------------------------------------------------------------------------------\n");
        printf("| Elementi   | N. Thread  | Tempo RELU   | Tempo MIN    | Tempo MAX    | Tempo FILL   | T. RESHAPE   |\n");
        printf("|------------|------------|--------------|--------------|--------------|--------------|--------------|\n");

        for (int t = 0; t < num_configs; t++) {
            int threads = thread_configs[t];
            if (threads <= max_threads) {
                run_bench(size, threads);
            }
        }
        if (max_threads != 1 && max_threads != 2 && max_threads != 4 && max_threads != 8) {
            run_bench(size, max_threads);
        }
        printf("\n");
    }

    return 0;
}
