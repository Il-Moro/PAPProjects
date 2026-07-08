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
		a->buffer->data[i] = (float)i;
		b->buffer->data[i] = (float)(i + 1);
	}

	omp_set_num_threads(num_threads);

	// Warm-up
	tensor *w1 = sum(a, b);
	tensorDeref(w1);

	// Benchmark SUM
	double sum_time = 0.0;
	for (int run = 0; run < NUM_RUNS; run++) {
		double start = omp_get_wtime();
		tensor *res = sum(a, b);
		double end = omp_get_wtime();
		sum_time += (end - start);
		tensorDeref(res);
	}

	// Benchmark SUB
	double sub_time = 0.0;
	for (int run = 0; run < NUM_RUNS; run++) {
		double start = omp_get_wtime();
		tensor *res = sub(a, b);
		double end = omp_get_wtime();
		sub_time += (end - start);
		tensorDeref(res);
	}

	// Benchmark MUL
	double mul_time = 0.0;
	for (int run = 0; run < NUM_RUNS; run++) {
		double start = omp_get_wtime();
		tensor *res = mul(a, b);
		double end = omp_get_wtime();
		mul_time += (end - start);
		tensorDeref(res);
	}

	double sum_ms = (sum_time / NUM_RUNS) * 1000.0;
	double sub_ms = (sub_time / NUM_RUNS) * 1000.0;
	double mul_ms = (mul_time / NUM_RUNS) * 1000.0;

	printf("| %10d | %10d | %12.4f ms | %12.4f ms | %12.4f ms |\n", size, num_threads, sum_ms, sub_ms, mul_ms);

	tensorDeref(a);
	tensorDeref(b);
}

int main() {
	printf("=== BENCHMARK DI SCALABILITÀ OPERAZIONI ARITMETICHE ===\n");
	printf("Numero massimo di thread disponibili: %d\n\n", omp_get_max_threads());

	int32_t sizes[] = {10000, 1000000, 10000000};
	int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

	int thread_configs[] = {1, 2, 4, 8};
	int num_configs = sizeof(thread_configs) / sizeof(thread_configs[0]);
	int max_threads = omp_get_max_threads();

	for (int s = 0; s < num_sizes; s++) {
		int32_t size = sizes[s];
		printf("------------------------------------------------------------------------------------\n");
		printf("Benchmark con Tensore di dimensione: %d elementi (%.2f MB per operando)\n", size, (double)size * sizeof(float) / 1024.0 / 1024.0);
		printf("------------------------------------------------------------------------------------\n");
		printf("| Elementi   | N. Thread  | Tempo SUM      | Tempo SUB      | Tempo MUL      |\n");
		printf("|------------|------------|----------------|----------------|----------------|\n");

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
