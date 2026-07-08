// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "tensor.h"

#define NUM_RUNS 5

void run_bench(int32_t size, int num_threads, double *time_1_thread) {
	int32_t shape_a[] = {size, size};
	int32_t shape_b[] = {size, size};
	tensor *a = allocTensor(2, shape_a);
	tensor *b = allocTensor(2, shape_b);

	// Inizializza con alcuni valori
	for (int32_t i = 0; i < size * size; i++) {
		a->buffer->data[i] = 1.0f;
		b->buffer->data[i] = 2.0f;
	}

	omp_set_num_threads(num_threads);

	// Warm-up
	tensor *warmup = matrixMul(a, b);
	tensorDeref(warmup);

	// Esegui il benchmark
	double total_time = 0.0;
	for (int run = 0; run < NUM_RUNS; run++) {
		double start = omp_get_wtime();
		tensor *res = matrixMul(a, b);
		double end = omp_get_wtime();
		
		total_time += (end - start);
		tensorDeref(res);
	}

	double avg_time_ms = (total_time / NUM_RUNS) * 1000.0;
	
	// Salva il tempo a 1 thread come baseline per calcolare lo speedup
	if (num_threads == 1) {
		*time_1_thread = avg_time_ms;
	}
	
	double speedup = *time_1_thread / avg_time_ms;
	double gflops = (2.0 * size * size * size) / (avg_time_ms / 1000.0) / 1e9;

	printf("| %10d | %12d | %12.4f ms | %11.2f GFLOPS | %8.2fx |\n", size, num_threads, avg_time_ms, gflops, speedup);

	tensorDeref(a);
	tensorDeref(b);
}

int main() {
	printf("=== BENCHMARK DI SCALABILITÀ MATRIX MULTIPLICATION (@) ===\n");
	printf("Numero massimo di thread fisici disponibili: %d\n\n", omp_get_max_threads());

	int32_t sizes[] = {100, 300, 700};
	int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

	int thread_configs[] = {1, 2, 4, 8};
	int num_configs = sizeof(thread_configs) / sizeof(thread_configs[0]);
	int max_threads = omp_get_max_threads();

	for (int s = 0; s < num_sizes; s++) {
		int32_t size = sizes[s];
		printf("--------------------------------------------------------------------------------\n");
		printf("Moltiplicazione di matrici quadrate: %d x %d (%.2f MB per operando)\n", size, size, (double)(size * size) * sizeof(float) / 1024.0 / 1024.0);
		printf("--------------------------------------------------------------------------------\n");
		printf("| Dimensione | N. Thread    | Tempo Medio (ms)  | GFLOPS      | Speedup  |\n");
		printf("|------------|--------------|-------------------|-------------|----------|\n");

		double time_1_thread = 1.0;
		for (int t = 0; t < num_configs; t++) {
			int threads = thread_configs[t];
			if (threads <= max_threads) {
				run_bench(size, threads, &time_1_thread);
			}
		}
		if (max_threads != 1 && max_threads != 2 && max_threads != 4 && max_threads != 8) {
			run_bench(size, max_threads, &time_1_thread);
		}
		printf("\n");
	}

	return 0;
}
