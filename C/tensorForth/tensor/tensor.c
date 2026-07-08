// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

// tensor.c - Lifecycle e utilità dei tensori:
//   allocTensor, tensorRef, tensorDeref, printTensor,
//   equalDimensions, equalShapes, getTotalElements

#include "tensor.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

// 1. Lifecycle: allocazione, referenziazione, dereferenziazione

// Alloca un nuovo tensore con struttura logica e buffer fisico separati.
// dimensionOfTensor: 1 per vettori, 2 per matrici.
// shape: array con le dimensioni (shape[0] = righe, shape[1] = colonne se 2D).
tensor *allocTensor(int32_t dimensionOfTensor, const int32_t *shape) {
	tensor *t = malloc(sizeof(tensor));
	if (t == NULL) {
		fprintf(stderr, "Impossibile allocare struct tensor\n");
		exit(1);
	}

	t->dimensionOfTensor = dimensionOfTensor;
	t->referenceCount = 1;
	t->shape[0] = shape[0];
	t->shape[1] = (dimensionOfTensor > 1) ? shape[1] : 1;

	t->buffer = malloc(sizeof(tensor_buffer));
	if (t->buffer == NULL) {
		free(t);
		fprintf(stderr, "Impossibile allocare struct tensor_buffer\n");
		exit(1);
	}

	t->buffer->referenceCount = 1;
	t->buffer->isMMapped = 0;
	t->buffer->mmapAddress = NULL;
	t->buffer->mmapLength = 0;

	int32_t size = shape[0];
	if (dimensionOfTensor > 1)
		size *= shape[1];

	int status = posix_memalign((void **)&(t->buffer->data), 64, (size_t)size * sizeof(float));
	if (status != 0) {
		free(t->buffer);
		free(t);
		fprintf(stderr, "Impossibile allocare float data (posix_memalign)\n");
		exit(1);
	}

	return t;
}

// Incrementa il reference count logico del tensore.
void tensorRef(tensor *t) {
	if (t != NULL) {
		t->referenceCount++;
	}
}

// Decrementa il reference count logico del tensore.
// Se raggiunge 0, decrementa anche il refcount del buffer fisico.
// Se anche il buffer raggiunge 0, libera i dati (munmap o free) e il buffer
// stesso.
void tensorDeref(tensor *t) {
	if (t == NULL)
		return;

	t->referenceCount--;
	if (t->referenceCount == 0) {
		if (t->buffer != NULL) {
			t->buffer->referenceCount--;
			if (t->buffer->referenceCount == 0) {
				if (t->buffer->isMMapped) {
					munmap(t->buffer->mmapAddress, t->buffer->mmapLength);
				} else {
					free(t->buffer->data);
				}
				free(t->buffer);
			}
		}
		free(t);
	}
}

// 2. Utilità

// Stampa il tensore nel formato richiesto dalla specifica
void printTensor(tensor *t) {
	if (t == NULL) {
		printf("Tensor(null)\n");
		return;
	}

	printf("Tensor(shape=[");
	if (t->dimensionOfTensor == 2) {
		printf("%d %d", t->shape[0], t->shape[1]);
	} else {
		printf("%d", t->shape[0]);
	}
	printf("], data=[");

	int32_t size = getTotalElements(t);
	for (int32_t i = 0; i < size; i++) {
		printf("%g", t->buffer->data[i]); // %g scegli automaticamente la rappresentazione più compatta e leggibile
		if (i < size - 1){
			printf(" ");
		}
	}
	printf("])\n");
}

// Verifica che le due dimensioni siano uguali; stampa errore ed esce
// altrimenti.
void equalDimensions(int dimensionA, int dimensionB) {
	if (dimensionA != dimensionB) {
		fprintf(stderr, "Errore: dimensioni incompatibili (%d vs %d)\n", dimensionA, dimensionB);
		exit(1);
	}
}

// Verifica che le shape siano uguali (a==b e c==d); stampa errore ed esce
// altrimenti.
void equalShapes(int a, int b, int c, int d) {
	if ((a != b) || (c != d)) {
		fprintf(stderr, "Errore: shape non compatibili (%d vs %d, %d vs %d)\n", a, b, c, d);
		exit(1);
	}
}

// Ritorna il numero totale di elementi del tensore (rows * cols per 2D, rows
// per 1D).
int32_t getTotalElements(tensor *t) {
	return (t->dimensionOfTensor == 2) ? (t->shape[0] * t->shape[1]) : t->shape[0];
}
