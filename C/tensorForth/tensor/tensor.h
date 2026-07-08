// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h> // size_t
#include <stdint.h> // int32_t
// TODO: da togliere (?)
#include <immintrin.h> // vettorizzazione -> forse non serve

typedef struct {
	int32_t referenceCount;
	int32_t isMMapped; // 32 + 32
	float *data;	   // 64
	void *mmapAddress; // 64
	size_t mmapLength; // 64
} tensor_buffer;	   // 256 bit

typedef struct {
	int32_t shape[2];		   // 64
	int32_t dimensionOfTensor; // 32
	int32_t referenceCount;	   // 32
	tensor_buffer *buffer;	   // 64
} tensor;					   // // 188 bit

// Struttura dell'header su disco per i file binari di tensori.
// MAX_DIM è fissato a 2. data_offset indica l'offset (in byte dall'inizio
// del file) dove iniziano i float, garantendo allineamento a 64 byte.
struct on_disk_tensor {
	int32_t shape[2];  // dimensioni del tensore (MAX_DIM = 2)
	int32_t ndim;	   // numero di dimensioni (1 o 2)
	off_t data_offset; // offset dei dati dall'inizio del file (= 64)
};

// OPERAZIONI:
// 1. Operazioni aritmetiche
tensor *sum(tensor *a, tensor *b); // +
tensor *sub(tensor *a, tensor *b); // -
tensor *mul(tensor *a, tensor *b); // *

// 2. Op. comparazione
tensor *equal(tensor *a, tensor *b);	   // =
tensor *greaterThan(tensor *a, tensor *b); // >
tensor *lessThan(tensor *a, tensor *b);	   // <

// 3. Op. Logiche
tensor *AND(tensor *a, tensor *b); // &
tensor *OR(tensor *a, tensor *b);  // |
tensor *NOT(tensor *a);			   // Operatore unario // !

// 4. Selezione
tensor *selection(tensor *a, tensor *b, tensor *mask); // Operatore ternario ($)

// 5. Op. specifiche per tensori
tensor *matrixMul(tensor *a, tensor *b);   // @
tensor *dotProduct(tensor *a, tensor *b);  // .
tensor *conv2D(tensor *a, tensor *kernel); // c

// 6. Op. di forma e Dimensione
tensor *reshape(tensor *t, tensor *shape); // r
tensor *ravel(tensor *t);				   // _
tensor *getShape(tensor *t);			   // #

// 7. Generazione, Riduzione e Riempimento
tensor *randomTensor(tensor *shape);	  // ?
tensor *relu(tensor *t);				  // R
tensor *minTensor(tensor *a, tensor *b);  // m
tensor *maxTensor(tensor *a, tensor *b);  // M
tensor *reductionSum(tensor *t);		  // S
tensor *fill(tensor *shape, tensor *val); // f

// 8. Lifecycle dei Tensori e Reference Counting
tensor *allocTensor(int32_t ndim, const int32_t *shape);
void tensorRef(tensor *t);
void tensorDeref(tensor *t);

// 9. Operazioni di I/O
tensor *readPGM(const char *fileName);			  // (
int writePGM(tensor *tensorToWrite, const char *fileName);	  // )
tensor *readBinary(const char *fileName);		  // {
int writeBinary(tensor *inputTensor, const char *fileName); // }

// 10. Utilità
void printTensor(tensor *t); // p

// ALTRO
void equalDimensions(int dimensionA, int dimensionB);
void equalShapes(int a, int b, int c, int d);
int32_t getTotalElements(tensor *t);

#endif