// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#ifndef TENSOR_H
#define TENSOR_H


#include <stdint.h> // int32_t
#include <stddef.h> // size_t
// TODO: da togliere (?)
#include <immintrin.h> // vettorizzazione -> forse non serve

typedef struct {
	int referenceCount;
	int isMMapped;
	float *data;	
	void *mmapAddress;
	size_t mmapLength;
} tensor_buffer;

typedef struct {
	int32_t shape[2];
	int32_t dimensionOfTensor;
	int referenceCount;
	tensor_buffer *buffer;	
} tensor;

// OPERAZIONI:
// 1. Operazioni aritmetiche
tensor *sum(tensor *a, tensor *b); // +
tensor *diff(tensor *a, tensor *b); // -
tensor *prod(tensor *a, tensor *b); // *

// 2. Op. comparazione
tensor *equal(tensor *a, tensor *b); // =
tensor *greaterThan(tensor *a, tensor *b); // <
tensor *lessThan(tensor *a, tensor *b); // >

// 3. Op. Logiche
tensor *AND(tensor *a, tensor *b); // &
tensor *OR(tensor *a, tensor *b); // \|
tensor *NOT(tensor *a); // Operatore unario // !

// 4. Selezione
tensor *selection(tensor *a, tensor *b, tensor *mask); // Operatore ternario ($)

// 5. Op. specifiche per tensori
tensor *matrixMul(tensor *a, tensor *b); // @
tensor *dotProduct(tensor *a, tensor *b); // .
tensor *conv2D(tensor *a, tensor *kernel); // c

// 6. Op. di forma e Dimensione
tensor *reshape(tensor *t, tensor *shape); // r
tensor *ravel(tensor *t); // _
tensor *getShape(tensor *t); // #

// 7. Generazione, Riduzione e Riempimento
tensor *randomTensor(tensor *shape); // ?
tensor *relu(tensor *t); // R
tensor *minTensor(tensor *a, tensor *b); // m
tensor *maxTensor(tensor *a, tensor *b); // M
tensor *reductionSum(tensor *t); // S
tensor *fill(tensor *shape, tensor *val); // f

// 8. Lifecycle dei Tensori e Reference Counting
tensor *allocTensor(int32_t ndim, const int32_t *shape);
void tensorRef(tensor *t);
void tensorDeref(tensor *t);

// 9. Operazioni di I/O
tensor *readPGM(const char *filename); // (
int writePGM(tensor *t, const char *filename); // )
tensor *readBinary(const char *filename); // {
int writeBinary(tensor *t, const char *filename); // }

// 10. Utilità
void printTensor(tensor *t); // p

#endif