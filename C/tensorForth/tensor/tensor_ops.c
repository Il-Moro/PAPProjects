// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

// tensor_ops.c - Tutte le operazioni sui tensori:
//   aritmetiche, comparazione, logiche, selezione,
//   algebriche (matmul, dot, conv2D), forma, generazione, riduzione, fill.

#include "tensor.h"
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Soglia di elementi oltre cui si attiva la parallelizzazione OpenMP
#define CONSTANT_FOR_PARALLIZATION 100000

// 1. Operazioni aritmetiche element-wise

// Somma elemento per elemento di a e b.
// Input: due tensori della stessa forma. Output: tensore risultante.
tensor *sum(tensor *a, tensor *b) {
	equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
	equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

	tensor *result = allocTensor(a->dimensionOfTensor, a->shape);
	int32_t totalElements = getTotalElements(a);
	float *dataOfA = a->buffer->data;
	float *dataOfB = b->buffer->data;
	float *dataResult = result->buffer->data;
	int32_t limit = totalElements - (totalElements % 8);

	#pragma omp parallel if (totalElements > CONSTANT_FOR_PARALLIZATION)
	{	
		#pragma omp for schedule(static) nowait
		for (int32_t i = 0; i < limit; i += 8) {
			__m256 va = _mm256_loadu_ps(&dataOfA[i]);
			__m256 vb = _mm256_loadu_ps(&dataOfB[i]);
			__m256 vres = _mm256_add_ps(va, vb);
			_mm256_storeu_ps(&dataResult[i], vres);
		}
	}

	for (int32_t i = limit; i < totalElements; i++) {
		dataResult[i] = dataOfA[i] + dataOfB[i];
	}

	return result;
}

// Differenza elemento per elemento (a - b).
// Input: due tensori della stessa forma. Output: tensore risultante.
tensor *sub(tensor *a, tensor *b) {
	equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
	equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

	tensor *result = allocTensor(a->dimensionOfTensor, a->shape);
	int32_t totalElements = getTotalElements(a);
	float *dataOfA = a->buffer->data;
	float *dataOfB = b->buffer->data;
	float *dataResult = result->buffer->data;
	int32_t limit = totalElements - (totalElements % 8);

	#pragma omp parallel if (totalElements > CONSTANT_FOR_PARALLIZATION)
	{
		#pragma omp for schedule(static) nowait
		for (int32_t i = 0; i < limit; i += 8) {
			__m256 va = _mm256_loadu_ps(&dataOfA[i]);
			__m256 vb = _mm256_loadu_ps(&dataOfB[i]);
			__m256 vres = _mm256_sub_ps(va, vb);
			_mm256_storeu_ps(&dataResult[i], vres);
		}
	}
	for (int32_t i = limit; i < totalElements; i++) {
		dataResult[i] = dataOfA[i] - dataOfB[i];
	}

	return result;
}

// Prodotto elemento per elemento (a * b).
// Input: due tensori della stessa forma. Output: tensore risultante.
tensor *mul(tensor *a, tensor *b) {
	equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
	equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

	tensor *result = allocTensor(a->dimensionOfTensor, a->shape);
	int32_t totalElements = getTotalElements(a);
	float *dataOfA = a->buffer->data;
	float *dataOfB = b->buffer->data;
	float *dataResult = result->buffer->data;
	int32_t limit = totalElements - (totalElements % 8);

	#pragma omp parallel if (totalElements > CONSTANT_FOR_PARALLIZATION)
	{
		#pragma omp for schedule(static) nowait
		for (int32_t i = 0; i < limit; i += 8) {
			__m256 va = _mm256_loadu_ps(&dataOfA[i]);
			__m256 vb = _mm256_loadu_ps(&dataOfB[i]);
			__m256 vres = _mm256_mul_ps(va, vb);
			_mm256_storeu_ps(&dataResult[i], vres);
		}
	}
	for (int32_t i = limit; i < totalElements; i++) {
		dataResult[i] = dataOfA[i] * dataOfB[i];
	}

	return result;
}

// 2. Operazioni di comparazione element-wise (risultato: 1.0f o 0.0f)

// OSS: le operazioni vettoriali SIMD del tipo _CMP_EQ_OQ non producono
// direttamente un vettore di 0f o 1f ma una maschera di bit, quindi valori del
// tipo 0x00000000 o 0xFFFFFFFF, per far sì di ottenere il risultato come valori
// float a 0f o 1f si usa la funzione blend(v1, v2, mask) che sulla base della
// mask, associa il bit più significativo di mask al valore corrispondente in v1
// o v2

// Uguaglianza elemento per elemento: result[i] = (a[i] == b[i]) ? 1.0f : 0.0f
tensor *equal(tensor *a, tensor *b) {
	equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
	equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

	tensor *result = allocTensor(a->dimensionOfTensor, a->shape);
	int32_t totalElements = getTotalElements(a);
	float *dataOfA = a->buffer->data;
	float *dataOfB = b->buffer->data;
	float *dataResult = result->buffer->data;
	int32_t limit = totalElements - (totalElements % 8);

	__m256 vTrue = _mm256_set1_ps(1.0f);
	__m256 vFalse = _mm256_setzero_ps();

	#pragma omp parallel if (totalElements > CONSTANT_FOR_PARALLIZATION)
	{
		#pragma omp for schedule(static) nowait
		for (int32_t i = 0; i < limit; i += 8) {
			__m256 va = _mm256_loadu_ps(&dataOfA[i]);
			__m256 vb = _mm256_loadu_ps(&dataOfB[i]);
			__m256 mask = _mm256_cmp_ps(va, vb, _CMP_EQ_OQ);	 // -> mask cmp CMP EQ OQ pone i valori a 0x000-- o 0xFFFF; OQ sta Ordered Quiet
			__m256 vres = _mm256_blendv_ps(vFalse, vTrue, mask); // -> blend sceglie da vFalse se mask è a 0x0000 e vTrue se è 0xFFF
			_mm256_storeu_ps(&dataResult[i], vres);
		}
	}
	for (int32_t i = limit; i < totalElements; i++) {
		dataResult[i] = (dataOfA[i] == dataOfB[i]) ? 1.0f : 0.0f;
	}

	return result;
}

// Maggiore di elemento per elemento: result[i] = (a[i] > b[i]) ? 1.0f : 0.0f
tensor *greaterThan(tensor *a, tensor *b) {
	equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
	equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

	tensor *result = allocTensor(a->dimensionOfTensor, a->shape);
	int32_t totalElements = getTotalElements(a);
	float *dataOfA = a->buffer->data;
	float *dataOfB = b->buffer->data;
	float *dataResult = result->buffer->data;
	int32_t limit = totalElements - (totalElements % 8);

	__m256 vTrue = _mm256_set1_ps(1.0f);
	__m256 vFalse = _mm256_setzero_ps();

	#pragma omp parallel if (totalElements > CONSTANT_FOR_PARALLIZATION)
	{
		#pragma omp for schedule(static) nowait
		for (int32_t i = 0; i < limit; i += 8) {
			__m256 va = _mm256_loadu_ps(&dataOfA[i]);
			__m256 vb = _mm256_loadu_ps(&dataOfB[i]);
			__m256 mask = _mm256_cmp_ps(va, vb, _CMP_GT_OQ);
			__m256 vres = _mm256_blendv_ps(vFalse, vTrue, mask);
			_mm256_storeu_ps(&dataResult[i], vres);
		}
	}
	for (int32_t i = limit; i < totalElements; i++) {
		dataResult[i] = (dataOfA[i] > dataOfB[i]) ? 1.0f : 0.0f;
	}

	return result;
}

// Minore di elemento per elemento: result[i] = (a[i] < b[i]) ? 1.0f : 0.0f
tensor *lessThan(tensor *a, tensor *b) {
	equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
	equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

	tensor *result = allocTensor(a->dimensionOfTensor, a->shape);
	int32_t totalElements = getTotalElements(a);
	float *dataOfA = a->buffer->data;
	float *dataOfB = b->buffer->data;
	float *dataResult = result->buffer->data;
	int32_t limit = totalElements - (totalElements % 8);

	__m256 vTrue = _mm256_set1_ps(1.0f);
	__m256 vFalse = _mm256_setzero_ps();

	#pragma omp parallel if (totalElements > CONSTANT_FOR_PARALLIZATION)
	{
		#pragma omp for schedule(static) nowait
		for (int32_t i = 0; i < limit; i += 8) {
			__m256 va = _mm256_loadu_ps(&dataOfA[i]);
			__m256 vb = _mm256_loadu_ps(&dataOfB[i]);
			__m256 mask = _mm256_cmp_ps(va, vb, _CMP_LT_OQ);
			__m256 vres = _mm256_blendv_ps(vFalse, vTrue, mask);
			_mm256_storeu_ps(&dataResult[i], vres);
		}
	}
	for (int32_t i = limit; i < totalElements; i++) {
		dataResult[i] = (dataOfA[i] < dataOfB[i]) ? 1.0f : 0.0f;
	}

	return result;
}

// 3. Operazioni logiche element-wise (input atteso: solo 0.0f e 1.0f)

// OSS: perchè usare doppio livello di mask -> per funzioni vettoriali non
// esistono direttamente le operazioni logiche, quindi fare un AND tra v1 e v2
// rappresentati vettorialmente in float, non produce valori booleani di 0f o
// 1f. Quindi si usa il confronto tra v0 e v1 con il vettore generando due
// vettori mask a cui è possibile a questo punto applicare il confronto diretto.
// Infine si usa sempre blend per separare e ottenere i valori di 0f e 1f

// AND logico: result[i] = (a[i]==1.0 && b[i]==1.0) ? 1.0f : 0.0f
tensor *AND(tensor *a, tensor *b) {
	equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
	equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

	tensor *result = allocTensor(a->dimensionOfTensor, a->shape);
	int32_t totalElements = getTotalElements(a);
	float *dataOfA = a->buffer->data;
	float *dataOfB = b->buffer->data;
	float *dataResult = result->buffer->data;
	int32_t limit = totalElements - (totalElements % 8);

	__m256 vTrue = _mm256_set1_ps(1.0f);
	__m256 vFalse = _mm256_setzero_ps();

	#pragma omp parallel if (totalElements > CONSTANT_FOR_PARALLIZATION)
	{
		#pragma omp for schedule(static) nowait
		for (int32_t i = 0; i < limit; i += 8) {
			__m256 va = _mm256_loadu_ps(&dataOfA[i]);
			__m256 vb = _mm256_loadu_ps(&dataOfB[i]);
			__m256 maskA = _mm256_cmp_ps(va, vTrue, _CMP_EQ_OQ);
			__m256 maskB = _mm256_cmp_ps(vb, vTrue, _CMP_EQ_OQ);
			__m256 maskRes = _mm256_and_ps(maskA, maskB);
			__m256 vres = _mm256_blendv_ps(vFalse, vTrue, maskRes);
			_mm256_storeu_ps(&dataResult[i], vres);
		}
	}
	for (int32_t i = limit; i < totalElements; i++) {
		dataResult[i] = ((dataOfA[i] == 1.0f) && (dataOfB[i] == 1.0f)) ? 1.0f : 0.0f;
	}

	return result;
}

// OR logico: result[i] = (a[i]==1.0 || b[i]==1.0) ? 1.0f : 0.0f
tensor *OR(tensor *a, tensor *b) {
	equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
	equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

	tensor *result = allocTensor(a->dimensionOfTensor, a->shape);
	int32_t totalElements = getTotalElements(a);
	float *dataOfA = a->buffer->data;
	float *dataOfB = b->buffer->data;
	float *dataResult = result->buffer->data;
	int32_t limit = totalElements - (totalElements % 8);

	__m256 vTrue = _mm256_set1_ps(1.0f);
	__m256 vFalse = _mm256_setzero_ps();

	#pragma omp parallel if (totalElements > CONSTANT_FOR_PARALLIZATION)
	{
		#pragma omp for schedule(static) nowait
		for (int32_t i = 0; i < limit; i += 8) {
			__m256 va = _mm256_loadu_ps(&dataOfA[i]);
			__m256 vb = _mm256_loadu_ps(&dataOfB[i]);
			__m256 maskA = _mm256_cmp_ps(va, vTrue, _CMP_EQ_OQ);
			__m256 maskB = _mm256_cmp_ps(vb, vTrue, _CMP_EQ_OQ);
			__m256 maskRes = _mm256_or_ps(maskA, maskB);
			__m256 vres = _mm256_blendv_ps(vFalse, vTrue, maskRes);
			_mm256_storeu_ps(&dataResult[i], vres);
		}
	}
	for (int32_t i = limit; i < totalElements; i++) {
		dataResult[i] = ((dataOfA[i] == 1.0f) || (dataOfB[i] == 1.0f)) ? 1.0f : 0.0f;
	}

	return result;
}

// NOT logico unario: result[i] = (a[i]==1.0) ? 0.0f : 1.0f
tensor *NOT(tensor *a) {
	tensor *result = allocTensor(a->dimensionOfTensor, a->shape);
	int32_t totalElements = getTotalElements(a);
	float *dataOfA = a->buffer->data;
	float *dataResult = result->buffer->data;
	int32_t limit = totalElements - (totalElements % 8);

	__m256 vTrue = _mm256_set1_ps(1.0f);
	__m256 vFalse = _mm256_setzero_ps();

	#pragma omp parallel if (totalElements > CONSTANT_FOR_PARALLIZATION)
	{
		#pragma omp for schedule(static) nowait
		for (int32_t i = 0; i < limit; i += 8) {
			__m256 va = _mm256_loadu_ps(&dataOfA[i]);
			__m256 mask = _mm256_cmp_ps(va, vTrue, _CMP_EQ_OQ);
			// Inverte: dove mask è 1 (era 1.0f) mette 0.0f, altrimenti 1.0f
			__m256 vres = _mm256_blendv_ps(vTrue, vFalse, mask);
			_mm256_storeu_ps(&dataResult[i], vres);
		}
	}
	for (int32_t i = limit; i < totalElements; i++) {
		dataResult[i] = (dataOfA[i] == 1.0f) ? 0.0f : 1.0f;
	}
	return result;
}

// 4. Selezione  ( b a m -- m?a:b )

// Selezione condizionale: result[i] = (mask[i]==1.0) ? a[i] : b[i]
// Input: a, b con stessa forma; mask di soli 0.0f e 1.0f.
tensor *selection(tensor *a, tensor *b, tensor *mask) {
	equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
	equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);
	equalDimensions(a->dimensionOfTensor, mask->dimensionOfTensor);
	equalShapes(a->shape[0], mask->shape[0], a->shape[1], mask->shape[1]);

	tensor *result = allocTensor(a->dimensionOfTensor, a->shape);
	int32_t totalElements = getTotalElements(a);
	float *dataOfA = a->buffer->data;
	float *dataOfB = b->buffer->data;
	float *dataMask = mask->buffer->data;
	float *dataResult = result->buffer->data;
	int32_t limit = totalElements - (totalElements % 8);

	__m256 vTrue = _mm256_set1_ps(1.0f);

	#pragma omp parallel if (totalElements > CONSTANT_FOR_PARALLIZATION)
	{
		#pragma omp for schedule(static) nowait
		for (int32_t i = 0; i < limit; i += 8) {
			__m256 va = _mm256_loadu_ps(&dataOfA[i]);
			__m256 vb = _mm256_loadu_ps(&dataOfB[i]);
			__m256 vmask = _mm256_loadu_ps(&dataMask[i]);
			__m256 bitMask = _mm256_cmp_ps(vmask, vTrue, _CMP_EQ_OQ); // virtual mask -> si ottiene una mask utilizzando vTrue e vmask attraverso comparazione di uguaglianza
			__m256 vres = _mm256_blendv_ps(vb, va, bitMask);		  // blend sceglie in base al bitmask
			_mm256_storeu_ps(&dataResult[i], vres);
		}
	}
	for (int32_t i = limit; i < totalElements; i++) {
		dataResult[i] = (dataMask[i] == 1.0f) ? dataOfA[i] : dataOfB[i];
	}

	return result;
}

// 5. Operazioni algebriche su tensori

// Prodotto matriciale A @ B (ordine IKJ).
// Input: a (MxK) e b (KxN) entrambe 2D. Output: matrice MxN.
tensor *matrixMul(tensor *a, tensor *b) {
	equalDimensions(a->dimensionOfTensor, 2);
	equalDimensions(b->dimensionOfTensor, 2);
	if (a->shape[1] != b->shape[0]) {
		fprintf(stderr, "Errore matrixMul: colonne di A (%d) != righe di B (%d)\n", a->shape[1], b->shape[0]);
		exit(1);
	}

	int32_t rowsA = a->shape[0];
	int32_t colsA = a->shape[1];
	int32_t colsB = b->shape[1];
	int32_t newShape[] = {rowsA, colsB};
	tensor *result = allocTensor(2, newShape);

	float *dataOfA = a->buffer->data;
	float *dataOfB = b->buffer->data;
	float *dataResult = result->buffer->data;
	int32_t size_C = rowsA * colsB;

	// Azzeramento del risultato
	#pragma omp parallel for schedule(static) if (size_C > CONSTANT_FOR_PARALLIZATION)
	for (int32_t i = 0; i < size_C; i++) {
		dataResult[i] = 0.0f;
	}

	int32_t limit = colsB - (colsB % 8);

	// Prodotto IKJ con FMA (fuse moltiply add): vettorizza lungo le colonne di B e C
	#pragma omp parallel for schedule(static) if ((rowsA * colsA * colsB) > 50000)
	for (int32_t i = 0; i < rowsA; i++) {
		for (int32_t k = 0; k < colsA; k++) {
			float aik = dataOfA[i * colsA + k];
			__m256 vaik = _mm256_set1_ps(aik);
			for (int32_t j = 0; j < limit; j += 8) {
				__m256 vb = _mm256_loadu_ps(&dataOfB[k * colsB + j]);
				__m256 vc = _mm256_loadu_ps(&dataResult[i * colsB + j]);
				__m256 vres = _mm256_fmadd_ps(vaik, vb, vc);
				_mm256_storeu_ps(&dataResult[i * colsB + j], vres);
			}
			for (int32_t j = limit; j < colsB; j++) {
				dataResult[i * colsB + j] += aik * dataOfB[k * colsB + j];
			}
		}
	}

	return result;
}

// Prodotto interno (dot product) tra due vettori 1D di uguale dimensione.
// Output: tensore 1D di un elemento contenente il valore scalare.
tensor *dotProduct(tensor *a, tensor *b) {
	equalDimensions(a->dimensionOfTensor, 1);
	equalDimensions(b->dimensionOfTensor, 1);
	if (a->shape[0] != b->shape[0]) {
		fprintf(stderr, "Errore dotProduct: dimensioni incompatibili (%d vs %d)\n", a->shape[0], b->shape[0]);
		exit(1);
	}

	int32_t size = a->shape[0];
	float *dataA = a->buffer->data;
	float *dataB = b->buffer->data;
	float total = 0.0;
	int32_t limit = size - (size % 8);

	#pragma omp parallel if (size > CONSTANT_FOR_PARALLIZATION)
	{
		float local = 0.0;
		
		#pragma omp for schedule(static) nowait
		for (int32_t i = 0; i < limit; i += 8) {
			__m256 va = _mm256_loadu_ps(&dataA[i]);
			__m256 vb = _mm256_loadu_ps(&dataB[i]);
			__m256 vm = _mm256_mul_ps(va, vb);
			float tmp[8];
			_mm256_storeu_ps(tmp, vm);
			local += (float)(tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]);
		}
		
		#pragma omp critical
		{
			total += local;
		}
	}
	for (int32_t i = limit; i < size; i++) {
		total += (float)(dataA[i] * dataB[i]);
	}

	int32_t shape[] = {1};
	tensor *result = allocTensor(1, shape);
	result->buffer->data[0] = (float)total;
	return result;
}

// Convoluzione 2D di 'a' con 'kernel'
// Output: tensore della stessa dimensione di 'a'.
// Ogni elemento dell'output è indipendente → parallelizzabile con OpenMP.
tensor *conv2D(tensor *a, tensor *kernel) {
	equalDimensions(a->dimensionOfTensor, 2);
	equalDimensions(kernel->dimensionOfTensor, 2);

	int32_t rowsA = a->shape[0];
	int32_t colsA = a->shape[1];
	int32_t rowsK = kernel->shape[0];
	int32_t colsK = kernel->shape[1];
	int32_t padR = (rowsK - 1) / 2;
	int32_t padC = (colsK - 1) / 2;

	int32_t shape[] = {rowsA, colsA};
	tensor *result = allocTensor(2, shape);
	float *dataA = a->buffer->data;
	float *dataK = kernel->buffer->data;
	float *dataRes = result->buffer->data;

	#pragma omp parallel for schedule(static) if ((rowsA * colsA) > CONSTANT_FOR_PARALLIZATION)
	for (int32_t i = 0; i < rowsA; i++) {
		int32_t j = 0;

		// --- 1. BLOCCO AVX UNROLLED (2x8 = 16 colonne alla volta) ---
		for (; j <= colsA - 16; j += 16) {
			__m256 sum0 = _mm256_setzero_ps();
			__m256 sum1 = _mm256_setzero_ps();

			for (int32_t ki = 0; ki < rowsK; ki++) {
				int32_t ii = i + ki - padR;
				if (ii < 0 || ii >= rowsA)
					continue;

				int32_t baseA = ii * colsA;

				for (int32_t kj = 0; kj < colsK; kj++) {
					float k_val = dataK[ki * colsK + kj];
					__m256 vk = _mm256_set1_ps(k_val);

					int32_t jj_start0 = j + kj - padC;
					int32_t jj_start1 = j + 8 + kj - padC;

					// Gestione Registro 0 (Colonne j ... j+7)
					if (jj_start0 >= 0 && jj_start0 + 7 < colsA) {
						__m256 va0 = _mm256_loadu_ps(&dataA[baseA + jj_start0]);
						sum0 = _mm256_fmadd_ps(va0, vk, sum0);
					} else {
						float temp_a0[8];
						for (int32_t elem = 0; elem < 8; elem++) {
							int32_t jj = jj_start0 + elem;
							temp_a0[elem] = (jj >= 0 && jj < colsA) ? dataA[baseA + jj] : 0.0f;
						}
						__m256 va0 = _mm256_loadu_ps(temp_a0);
						sum0 = _mm256_fmadd_ps(va0, vk, sum0);
					}

					// Gestione Registro 1 (Colonne j+8 ... j+15)
					if (jj_start1 >= 0 && jj_start1 + 7 < colsA) {
						__m256 va1 = _mm256_loadu_ps(&dataA[baseA + jj_start1]);
						sum1 = _mm256_fmadd_ps(va1, vk, sum1);
					} else {
						float temp_a1[8];
						for (int32_t elem = 0; elem < 8; elem++) {
							int32_t jj = jj_start1 + elem;
							temp_a1[elem] = (jj >= 0 && jj < colsA) ? dataA[baseA + jj] : 0.0f;
						}
						__m256 va1 = _mm256_loadu_ps(temp_a1);
						sum1 = _mm256_fmadd_ps(va1, vk, sum1);
					}
				}
			}
			_mm256_storeu_ps(&dataRes[i * colsA + j], sum0);
			_mm256_storeu_ps(&dataRes[i * colsA + j + 8], sum1);
		}

		// --- 2. CODA VETTORIALE SINGOLA (Se rimangono tra gli 8 e i 15 elementi) ---
		for (; j <= colsA - 8; j += 8) {
			__m256 sum_single = _mm256_setzero_ps();

			for (int32_t ki = 0; ki < rowsK; ki++) {
				int32_t ii = i + ki - padR;
				if (ii < 0 || ii >= rowsA)
					continue;

				int32_t baseA = ii * colsA;

				for (int32_t kj = 0; kj < colsK; kj++) {
					float k_val = dataK[ki * colsK + kj];
					__m256 vk = _mm256_set1_ps(k_val);

					int32_t jj_start = j + kj - padC;

					if (jj_start >= 0 && jj_start + 7 < colsA) {
						__m256 va = _mm256_loadu_ps(&dataA[baseA + jj_start]);
						sum_single = _mm256_fmadd_ps(va, vk, sum_single);
					} else {
						float temp_a[8];
						for (int32_t elem = 0; elem < 8; elem++) {
							int32_t jj = jj_start + elem;
							temp_a[elem] = (jj >= 0 && jj < colsA) ? dataA[baseA + jj] : 0.0f;
						}
						__m256 va = _mm256_loadu_ps(temp_a);
						sum_single = _mm256_fmadd_ps(va, vk, sum_single);
					}
				}
			}
			_mm256_storeu_ps(&dataRes[i * colsA + j], sum_single);
		}

		// --- 3. CODA SCALARE (Per gli ultimi residui stretti, < 8 elementi) ---
		for (; j < colsA; j++) {
			float s = 0.0f;
			for (int32_t ki = 0; ki < rowsK; ki++) {
				int32_t ii = i + ki - padR;
				if (ii < 0 || ii >= rowsA)
					continue;

				int32_t baseA = ii * colsA;

				for (int32_t kj = 0; kj < colsK; kj++) {
					int32_t jj = j + kj - padC;
					if (jj >= 0 && jj < colsA) {
						s += dataA[baseA + jj] * dataK[ki * colsK + kj];
					}
				}
			}
			dataRes[i * colsA + j] = s;
		}
	}
	return result;
}

// 6. Operazioni di forma e dimensione

// Reshape: cambia la forma di 't' secondo il tensore 'shapeTensor' (1D di 1 o 2
// elem). Non copia i dati: condivide il buffer fisico incrementando il suo
// refcount.
tensor *reshape(tensor * t, tensor * shapeTensor) {
	float *shapeData = shapeTensor->buffer->data;
	int32_t numDims = getTotalElements(shapeTensor);

	int32_t newTotal = (int32_t)shapeData[0];
	if (numDims == 2)
		newTotal *= (int32_t)shapeData[1];

	if (newTotal != getTotalElements(t)) {
		fprintf(stderr, "Errore reshape: shape non compatibile (%d elementi vs %d)\n", newTotal, getTotalElements(t));
		exit(1);
	}

	tensor *result = malloc(sizeof(tensor));
	if (result == NULL) {
		fprintf(stderr, "Errore: allocazione tensor\n");
		exit(1);
	}

	result->dimensionOfTensor = numDims;
	result->referenceCount = 1;
	result->shape[0] = (int32_t)shapeData[0];
	result->shape[1] = (numDims == 2) ? (int32_t)shapeData[1] : 1;

	// Condivisione del buffer fisico
	result->buffer = t->buffer;
	if (result->buffer != NULL)
		result->buffer->referenceCount++;

	return result;
}

// Ravel: appiattisce il tensore in un vettore 1D senza copiare i dati.
// Condivide il buffer fisico incrementando il suo refcount.
tensor *ravel(tensor * t) {
	tensor *result = malloc(sizeof(tensor));
	if (result == NULL) {
		fprintf(stderr, "Errore: allocazione tensor\n");
		exit(1);
	}

	result->dimensionOfTensor = 1;
	result->referenceCount = 1;
	result->shape[0] = getTotalElements(t);
	result->shape[1] = 1;

	result->buffer = t->buffer;
	if (result->buffer != NULL)
		result->buffer->referenceCount++;

	return result;
}

// Ritorna la forma di 't' come tensore 1D (1 elemento per vettori, 2 per
// matrici).
tensor *getShape(tensor * t) {
	int32_t shapeSize = t->dimensionOfTensor;
	int32_t resultShape[] = {shapeSize};
	tensor *result = allocTensor(1, resultShape);

	result->buffer->data[0] = (float)t->shape[0];
	if (shapeSize == 2)
		result->buffer->data[1] = (float)t->shape[1];

	return result;
}

// 7. Generazione, riduzione e riempimento

// Genera un tensore della forma indicata da 'shape' (tensore 1D di 1 o 2 elem)
// riempito con float casuali in [0, 1]. Non parallelizzato (rand() non
// thread-safe).
tensor *randomTensor(tensor * shape) {
	equalDimensions(shape->dimensionOfTensor, 1);

	float *shapeData = shape->buffer->data;
	int32_t numDims = shape->shape[0];
	int32_t newShape[2] = {(int32_t)shapeData[0], 1};
	if (numDims == 2)
		newShape[1] = (int32_t)shapeData[1];

	int32_t n = newShape[0] * newShape[1];
	tensor *result = allocTensor(numDims, newShape);

	for (int32_t i = 0; i < n; i++)
		result->buffer->data[i] = (float)rand() / (float)RAND_MAX;

	return result;
}

// ReLU: result[i] = max(0, t[i]).
// Input: qualsiasi tensore. Output: tensore della stessa forma.
tensor *relu(tensor * t) {
	int32_t size = getTotalElements(t);
	int32_t shape[] = {t->shape[0], t->shape[1]};
	tensor *result = allocTensor(t->dimensionOfTensor, shape);
	float *dataT = t->buffer->data;
	float *dataResult = result->buffer->data;
	int32_t limit = size - (size % 8);
	__m256 vzero = _mm256_setzero_ps();

	#pragma omp parallel for schedule(static) if (size > CONSTANT_FOR_PARALLIZATION)
	for (int32_t i = 0; i < limit; i += 8) {
		__m256 vt = _mm256_loadu_ps(&dataT[i]);
		__m256 vres = _mm256_max_ps(vzero, vt);
		_mm256_storeu_ps(&dataResult[i], vres);
	}
	for (int32_t i = limit; i < size; i++)
		dataResult[i] = (dataT[i] > 0.0f) ? dataT[i] : 0.0f;

	return result;
}

// Minimo elemento per elemento tra a e b.
// Input: due tensori della stessa forma. Output: tensore risultante.
tensor *minTensor(tensor * a, tensor * b) {
	equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

	int32_t size = getTotalElements(a);
	int32_t shape[] = {a->shape[0], a->shape[1]};
	tensor *result = allocTensor(a->dimensionOfTensor, shape);
	float *dataA = a->buffer->data;
	float *dataB = b->buffer->data;
	float *dataResult = result->buffer->data;
	int32_t limit = size - (size % 8);

	#pragma omp parallel for schedule(static) if (size > CONSTANT_FOR_PARALLIZATION)
	for (int32_t i = 0; i < limit; i += 8) {
		__m256 va = _mm256_loadu_ps(&dataA[i]);
		__m256 vb = _mm256_loadu_ps(&dataB[i]);
		__m256 vres = _mm256_min_ps(va, vb);
		_mm256_storeu_ps(&dataResult[i], vres);
	}
	for (int32_t i = limit; i < size; i++)
		dataResult[i] = (dataA[i] < dataB[i]) ? dataA[i] : dataB[i];

	return result;
}

// Massimo elemento per elemento tra a e b.
// Input: due tensori della stessa forma. Output: tensore risultante.
tensor *maxTensor(tensor * a, tensor * b) {
	equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
	equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

	int32_t size = getTotalElements(a);
	int32_t shape[] = {a->shape[0], a->shape[1]};
	tensor *result = allocTensor(a->dimensionOfTensor, shape);
	float *dataA = a->buffer->data;
	float *dataB = b->buffer->data;
	float *dataResult = result->buffer->data;
	int32_t limit = size - (size % 8);

	#pragma omp parallel for schedule(static) if (size > CONSTANT_FOR_PARALLIZATION)
	for (int32_t i = 0; i < limit; i += 8) {
		__m256 va = _mm256_loadu_ps(&dataA[i]);
		__m256 vb = _mm256_loadu_ps(&dataB[i]);
		__m256 vres = _mm256_max_ps(va, vb);
		_mm256_storeu_ps(&dataResult[i], vres);
	}
	for (int32_t i = limit; i < size; i++)
		dataResult[i] = (dataA[i] > dataB[i]) ? dataA[i] : dataB[i];

	return result;
}

// Somma di riduzione: somma tutti gli elementi di t.
// Input: qualsiasi tensore (1D o 2D). Output: tensore 1D di un elemento.
tensor *reductionSum(tensor * t) {
	// Nota: S funziona su qualsiasi tensore (1D o 2D) secondo la specifica
	int32_t size = getTotalElements(t);
	float *dataT = t->buffer->data;
	double total = 0.0;
	int32_t limit = size - (size % 8);

	#pragma omp parallel if (size > CONSTANT_FOR_PARALLIZATION)
	{
		__m256 vsum = _mm256_setzero_ps();

		#pragma omp for schedule(static) nowait
		for (int32_t i = 0; i < limit; i += 8) {
			__m256 vt = _mm256_loadu_ps(&dataT[i]);
			vsum = _mm256_add_ps(vsum, vt);
		}

		float tmp[8];
		_mm256_storeu_ps(tmp, vsum);

		double local = (double)(tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]);

		#pragma omp critical
		{
			total += local; // Accumulo finale sicuro nel double globale
		}
	}

	for (int32_t i = limit; i < size; i++)
		total += (double)dataT[i];

	int32_t shape[] = {1};
	tensor *result = allocTensor(1, shape);
	result->buffer->data[0] = (float)total;
	return result;
}

// Fill: crea un tensore di forma 'shape' riempito con i valori di 'val'
// ripetuti. shape: tensore 1D di 1 o 2 elementi. val: tensore 1D dei valori da
// ripetere.
tensor *fill(tensor * shape, tensor * val) {
	equalDimensions(shape->dimensionOfTensor, 1);
	equalDimensions(val->dimensionOfTensor, 1);

	float *shapeData = shape->buffer->data;
	int32_t dimResult = shape->shape[0];
	int32_t resShape[2];
	resShape[0] = (int32_t)shapeData[0];
	resShape[1] = (dimResult == 2) ? (int32_t)shapeData[1] : 1;

	int32_t total = resShape[0] * resShape[1];
	tensor *result = allocTensor(dimResult, resShape);
	float *dataVal = val->buffer->data;
	int32_t sizeVal = getTotalElements(val);
	float *dataResult = result->buffer->data;

	#pragma omp parallel for schedule(static) if (total > CONSTANT_FOR_PARALLIZATION)
	for (int32_t i = 0; i < total; i++)
		dataResult[i] = dataVal[i % sizeVal];

	return result;
}

