// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "tensor.h"

#define NUM_RUNS 5

void run_bench_conv(int32_t imgSize, int32_t kernSize, int num_threads) {
    int32_t shape_img[] = {imgSize, imgSize};
    int32_t shape_ker[] = {kernSize, kernSize};
    tensor *img = allocTensor(2, shape_img);
    tensor *ker = allocTensor(2, shape_ker);

    int32_t n_img = imgSize * imgSize;
    int32_t n_ker = kernSize * kernSize;
    for (int32_t i = 0; i < n_img; i++) img->buffer->data[i] = (float)rand() / RAND_MAX;
    for (int32_t i = 0; i < n_ker; i++) ker->buffer->data[i] = 1.0f / (float)n_ker;

    omp_set_num_threads(num_threads);

    // Warm-up
    tensor *w = conv2D(img, ker);
    tensorDeref(w);

    double total = 0.0;
    for (int run = 0; run < NUM_RUNS; run++) {
        double t0 = omp_get_wtime();
        tensor *res = conv2D(img, ker);
        total += omp_get_wtime() - t0;
        tensorDeref(res);
    }

    printf("| %8dx%-8d | %7dx%-7d | %10d | %12.4f ms |\n",
           imgSize, imgSize, kernSize, kernSize,
           num_threads, (total / NUM_RUNS) * 1000.0);

    tensorDeref(img);
    tensorDeref(ker);
}

void run_bench_dot(int32_t size, int num_threads) {
    int32_t shape[] = {size};
    tensor *a = allocTensor(1, shape);
    tensor *b = allocTensor(1, shape);
    for (int32_t i = 0; i < size; i++) {
        a->buffer->data[i] = (float)i;
        b->buffer->data[i] = 1.0f;
    }

    omp_set_num_threads(num_threads);

    tensor *w = dotProduct(a, b);
    tensorDeref(w);

    double total = 0.0;
    for (int run = 0; run < NUM_RUNS; run++) {
        double t0 = omp_get_wtime();
        tensor *res = dotProduct(a, b);
        total += omp_get_wtime() - t0;
        tensorDeref(res);
    }
    printf("| %10d | %10d | %14.4f ms |\n",
           size, num_threads, (total / NUM_RUNS) * 1000.0);

    tensorDeref(a);
    tensorDeref(b);
}

void run_bench_reductionSum(int32_t size, int num_threads) {
    int32_t shape[] = {size};
    tensor *t = allocTensor(1, shape);
    for (int32_t i = 0; i < size; i++) t->buffer->data[i] = 1.0f;

    omp_set_num_threads(num_threads);

    tensor *w = reductionSum(t);
    tensorDeref(w);

    double total = 0.0;
    for (int run = 0; run < NUM_RUNS; run++) {
        double t0 = omp_get_wtime();
        tensor *res = reductionSum(t);
        total += omp_get_wtime() - t0;
        tensorDeref(res);
    }
    printf("| %10d | %10d | %14.4f ms |\n",
           size, num_threads, (total / NUM_RUNS) * 1000.0);

    tensorDeref(t);
}

int main() {
    int thread_configs[] = {1, 2, 4, 8};
    int num_configs = sizeof(thread_configs) / sizeof(thread_configs[0]);
    int max_threads = omp_get_max_threads();

    printf("=== BENCHMARK CONVOLUZIONE 2D (conv2D) ===\n");
    printf("Thread disponibili: %d\n\n", max_threads);
    printf("| Immagine          | Kernel            | N. Thread  | Tempo medio    |\n");
    printf("|-------------------|-------------------|------------|----------------|\n");
    int32_t img_sizes[]  = {256, 512, 1024};
    int32_t ker_sizes[]  = {3, 5, 7};
    for (int s = 0; s < 3; s++) {
        for (int k = 0; k < 3; k++) {
            for (int t = 0; t < num_configs; t++) {
                if (thread_configs[t] <= max_threads)
                    run_bench_conv(img_sizes[s], ker_sizes[k], thread_configs[t]);
            }
        }
        printf("\n");
    }

    printf("\n=== BENCHMARK DOT PRODUCT (.) ===\n");
    printf("| Elementi   | N. Thread  | Tempo medio    |\n");
    printf("|------------|------------|----------------|\n");
    int32_t dot_sizes[] = {100000, 1000000, 10000000};
    for (int s = 0; s < 3; s++) {
        for (int t = 0; t < num_configs; t++) {
            if (thread_configs[t] <= max_threads)
                run_bench_dot(dot_sizes[s], thread_configs[t]);
        }
        printf("\n");
    }

    printf("\n=== BENCHMARK REDUCTION SUM (S) ===\n");
    printf("| Elementi   | N. Thread  | Tempo medio    |\n");
    printf("|------------|------------|----------------|\n");
    int32_t rs_sizes[] = {100000, 1000000, 10000000};
    for (int s = 0; s < 3; s++) {
        for (int t = 0; t < num_configs; t++) {
            if (thread_configs[t] <= max_threads)
                run_bench_reductionSum(rs_sizes[s], thread_configs[t]);
        }
        printf("\n");
    }

    return 0;
}
