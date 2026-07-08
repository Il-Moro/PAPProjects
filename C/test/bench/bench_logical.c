// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../../tensorForth/tensor.h"

#define NUM_RUNS 10

void run_bench(int32_t size, int num_threads) {
	int32_t shape[] = {size};
	tensor *a = allocTensor(1, shape);
	tensor *b = allocTensor(1, shape);

	for (int32_t i = 0; i < size; i++) {
		a->buffer->data[i] = (i % 2 == 0) ? 1.0f : 0.0f;
		b->buffer->data[i] = (i % 3 == 0) ? 1.0f : 0.0f;
	}

	omp_set_num_threads(num_threads);

	// Warm-up
	tensor *w1 = AND(a, b);
	tensorDeref(w1);

	// Benchmark AND
	double and_time = 0.0;
	for (int run = 0; run < NUM_RUNS; run++) {
		double start = omp_get_wtime();
		tensor *res = AND(a, b);
		double end = omp_get_wtime();
		and_time += (end - start);
		tensorDeref(res);
	}

	// Benchmark OR
	double or_time = 0.0;
	for (int run = 0; run < NUM_RUNS; run++) {
		double start = omp_get_wtime();
		tensor *res = OR(a, b);
		double end = omp_get_wtime();
		or_time += (end - start);
		tensorDeref(res);
	}

	// Benchmark NOT
	double not_time = 0.0;
	for (int run = 0; run < NUM_RUNS; run++) {
		double start = omp_get_wtime();
		tensor *res = NOT(a);
		double end = omp_get_wtime();
		not_time += (end - start);
		tensorDeref(res);
	}

	double and_ms = (and_time / NUM_RUNS) * 1000.0;
	double or_ms = (or_time / NUM_RUNS) * 1000.0;
	double not_ms = (not_time / NUM_RUNS) * 1000.0;

	printf("| %10d | %10d | %12.4f ms | %12.4f ms | %12.4f ms |\n", size, num_threads, and_ms, or_ms, not_ms);

	tensorDeref(a);
	tensorDeref(b);
}

int main() {
	printf("=== BENCHMARK DI SCALABILITÀ OPERAZIONI LOGICHE ===\n");
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
		printf("| Elementi   | N. Thread  | Tempo AND      | Tempo OR       | Tempo NOT      |\n");
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
