// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../tensorForth/tensor.h"

#define NUM_RUNS 10

void run_benchmark(int32_t size, int num_threads) {
	int32_t shape[] = {size};
	tensor *a = allocTensor(1, shape);
	tensor *b = allocTensor(1, shape);

	// Inizializza con alcuni valori
	for (int32_t i = 0; i < size; i++) {
		a->buffer->data[i] = 1.0f;
		b->buffer->data[i] = 2.0f;
	}

	// Imposta il numero di thread per OpenMP
	omp_set_num_threads(num_threads);

	// Warm-up per evitare l'influenza di cold cache e thread creation overhead
	tensor *warmup = sum(a, b);
	tensorDeref(warmup);

	// Esegui il benchmark
	double total_time = 0.0;
	for (int run = 0; run < NUM_RUNS; run++) {
		double start = omp_get_wtime();
		tensor *res = sum(a, b);
		double end = omp_get_wtime();
		
		total_time += (end - start);
		tensorDeref(res);
	}

	double avg_time_ms = (total_time / NUM_RUNS) * 1000.0;
	double throughput = (double)size / (avg_time_ms / 1000.0) / 1e6; // MOPs/sec (Milioni di operazioni al secondo)

	printf("| %10d | %12d | %12.4f ms | %15.2f MOPs/s |\n", size, num_threads, avg_time_ms, throughput);

	tensorDeref(a);
	tensorDeref(b);
}

int main() {
	printf("=== BENCHMARK DI SCALABILITÀ TENSOR.C ===\n");
	printf("Numero massimo di thread fisici disponibili: %d\n\n", omp_get_max_threads());

	int32_t sizes[] = {10000, 100000, 1000000, 10000000};
	int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

	int thread_configs[] = {1, 2, 4, 8};
	int num_configs = sizeof(thread_configs) / sizeof(thread_configs[0]);
	int max_threads = omp_get_max_threads();

	for (int s = 0; s < num_sizes; s++) {
		int32_t size = sizes[s];
		printf("-----------------------------------------------------------------------\n");
		printf("Benchmark con Tensore di dimensione: %d elementi (%.2f MB)\n", size, (double)size * sizeof(float) / 1024.0 / 1024.0);
		printf("-----------------------------------------------------------------------\n");
		printf("| Elementi   | N. Thread    | Tempo Medio (ms)  | Prestazioni (MOPs/s) |\n");
		printf("|------------|--------------|-------------------|----------------------|\n");

		for (int t = 0; t < num_configs; t++) {
			int threads = thread_configs[t];
			if (threads <= max_threads) {
				run_benchmark(size, threads);
			}
		}
		// Esegui anche con il massimo dei thread disponibili se non già fatto
		if (max_threads != 1 && max_threads != 2 && max_threads != 4 && max_threads != 8) {
			run_benchmark(size, max_threads);
		}
		printf("\n");
	}

	return 0;
}
