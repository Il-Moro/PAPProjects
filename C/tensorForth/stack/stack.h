// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#ifndef STACK_H
#define STACK_H

#include "../tensor/tensor.h"

typedef enum {
    TYPE_TENSOR,
    TYPE_STRING
} elementType;

typedef struct {
    elementType elementType;
    union {
        tensor *tensor;
        char *fileName;
    } value;
} stackElement;

typedef struct {
    stackElement *elements;
    int32_t index;
    int32_t capacityAllocated;
} stack;

// OPERAZIONI:

// 1. Inizializzazione e Distruzione
stack *stackInit(int32_t initialCapacity);
void stackDestroy(stack *stackPtr);

// 2. Operazioni di Push e Pop
void pushTensor(stack *stackPtr, tensor *inputTensor);
void pushFileName(stack *stackPtr, const char *fileName);
stackElement pop(stack *stackPtr);
stackElement peek(stack *stackPtr, int32_t stackOffset); // 0 = cima, 1 = sotto la cima, ecc.

// 3. Operatori dello stack (TensorForth)
int32_t stackDup(stack *stackPtr);   // d: duplica la cima
int32_t swap(stack *stackPtr);  // s: scambia i due elementi in cima
int32_t over(stack *stackPtr);  // o: copia il secondo elemento in cima
int32_t drop(stack *stackPtr);  // D: rimuove l'elemento in cima

// 4. Utilità
int32_t getStackSize(stack *stackPtr);

#endif