// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#ifndef STACK_H
#define STACK_H

#include "tensor.h"

typedef enum{
    TYPE_TENSOR,
    TYPE_STRING
} elementType;


typedef struct{
    elementType elementType;
    union{
        tensor* tensor;
        char* fileName;
    } value;
} stackElement;

typedef struct{
    stackElement *elements;
    int32_t index;
    int32_t capacityAllocated;
     
} stack;

// OPERAZIONI:

// 1. Inizializzazione e Distruzione
stack *stack_init(int32_t initial_capacity);
void stack_destroy(stack *s);

// 2. Operazioni di Push e Pop
void pushTensor(stack *s, tensor *t);
void pushFileName(stack *s, const char *name);
stackElement pop(stack *s);
stackElement peek(stack *s, int32_t offset); // 0 = cima, 1 = sotto la cima, ecc.

// 3. Operatori dello stack (TensorForth)
int32_t dup(stack *s);   // d: duplica la cima
int32_t swap(stack *s);  // s: scambia i due elementi in cima
int32_t over(stack *s);  // o: copia il secondo elemento in cima
int32_t drop(stack *s);  // D: rimuove l'elemento in cima

// 4. Utilità
int32_t getStackSize(stack *s);


#endif