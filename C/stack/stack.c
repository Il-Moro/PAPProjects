// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#include "stack.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STACK_GROWTH_FACTOR 2

// 1. Inizializzazione e Distruzione

stack *stackInit(int32_t initialCapacity) {
    if (initialCapacity <= 0) initialCapacity = 8;

    stack *stackPtr = malloc(sizeof(stack));
    if (stackPtr == NULL) {
        fprintf(stderr, "Errore: impossibile allocare lo stack\n");
        exit(1);
    }

    stackPtr->elements = malloc(sizeof(stackElement) * (size_t)initialCapacity);
    if (stackPtr->elements == NULL) {
        free(stackPtr);
        fprintf(stderr, "Errore: impossibile allocare gli elementi dello stack\n");
        exit(1);
    }

    stackPtr->index = 0;                     
    stackPtr->capacityAllocated = initialCapacity;
    return stackPtr;
}

void stackDestroy(stack *stackPtr) {
    if (stackPtr == NULL) return;

    // Libera tutti gli elementi rimasti
    for (int32_t i = 0; i < stackPtr->index; i++) {
        if (stackPtr->elements[i].elementType == TYPE_TENSOR) {
            tensorDeref(stackPtr->elements[i].value.tensor);
        } else {
            free(stackPtr->elements[i].value.fileName);
        }
    }

    free(stackPtr->elements);
    free(stackPtr);
}

// Funzione interna: crescita dinamica dello stack
static void stackGrow(stack *stackPtr) {
    int32_t newCapacity = stackPtr->capacityAllocated * STACK_GROWTH_FACTOR;
    stackElement *newElements = realloc(stackPtr->elements, sizeof(stackElement) * (size_t)newCapacity);
    if (newElements == NULL) {
        fprintf(stderr, "Errore: impossibile riallocare lo stack\n");
        exit(1);
    }
    stackPtr->elements = newElements;
    stackPtr->capacityAllocated = newCapacity;
}

// 2. Operazioni di Push e Pop 

void pushTensor(stack *stackPtr, tensor *inputTensor) {
    if (stackPtr->capacityAllocated <= stackPtr->index) {
        stackGrow(stackPtr);
    }
    tensorRef(inputTensor); // Incrementa il refcount logico: lo stack è un owner
    stackPtr->elements[stackPtr->index].elementType = TYPE_TENSOR;
    stackPtr->elements[stackPtr->index].value.tensor = inputTensor;
    stackPtr->index++;
}

void pushFileName(stack *stackPtr, const char *fileName) {
    if (stackPtr->index >= stackPtr->capacityAllocated) {
        stackGrow(stackPtr);
    }
    char *fileNameCopy = strdup(fileName);
    if (fileNameCopy == NULL) {
        fprintf(stderr, "Errore: impossibile duplicare la stringa\n");
        exit(1);
    }
    stackPtr->elements[stackPtr->index].elementType = TYPE_STRING;
    stackPtr->elements[stackPtr->index].value.fileName = fileNameCopy;
    stackPtr->index++;
}

// Pop: rimuove e restituisce l'elemento in cima.
// NOTA: per i tensori NON decrementa il refcount; il chiamante diventa owner.
// Per le stringhe, il chiamante è responsabile della free() del puntatore.
stackElement pop(stack *stackPtr) {
    if (stackPtr->index <= 0) {
        fprintf(stderr, "Errore: pop su stack vuoto\n");
        exit(1);
    }
    stackPtr->index--;
    return stackPtr->elements[stackPtr->index];
}

// Peek: ispeziona senza rimuovere. offset 0 = cima, 1 = sotto la cima, ecc.
// Restituisce per VALORE (copia shallow dell'elemento). NON modifica i refcount.
stackElement peek(stack *stackPtr, int32_t stackOffset) {
    int32_t targetIndex = stackPtr->index - 1 - stackOffset;
    if (targetIndex < 0) {
        fprintf(stderr, "Errore: peek oltre il fondo dello stack (offset=%d, size=%d)\n",
                stackOffset, stackPtr->index);
        exit(1);
    }
    return stackPtr->elements[targetIndex];
}

// 3. Operatori dello stack (TensorForth)

// d: duplica la cima. Per i tensori, copia il puntatore e incrementa il refcount.
int32_t stackDup(stack *stackPtr) {
    if (stackPtr->index <= 0) {
        fprintf(stderr, "Errore: stackDup su stack vuoto\n");
        exit(1);
    }
    stackElement topElement = stackPtr->elements[stackPtr->index - 1];
    if (topElement.elementType == TYPE_TENSOR) {
        pushTensor(stackPtr, topElement.value.tensor);   // pushTensor chiama tensorRef internamente
    } else {
        pushFileName(stackPtr, topElement.value.fileName); // pushFileName duplica la stringa
    }
    return 0;
}

// s: scambia i due elementi in cima. Non altera i refcount.
int32_t swap(stack *stackPtr) {
    if (stackPtr->index < 2) {
        fprintf(stderr, "Errore: swap richiede almeno 2 elementi sullo stack\n");
        exit(1);
    }
    stackElement tempElement = stackPtr->elements[stackPtr->index - 1];
    stackPtr->elements[stackPtr->index - 1] = stackPtr->elements[stackPtr->index - 2];
    stackPtr->elements[stackPtr->index - 2] = tempElement;
    return 0;
}

// o: copia il secondo elemento e lo inserisce in cima. Incrementa il refcount del tensore.
int32_t over(stack *stackPtr) {
    if (stackPtr->index < 2) {
        fprintf(stderr, "Errore: over richiede almeno 2 elementi sullo stack\n");
        exit(1);
    }
    stackElement secondElement = stackPtr->elements[stackPtr->index - 2];
    if (secondElement.elementType == TYPE_TENSOR) {
        pushTensor(stackPtr, secondElement.value.tensor);
    } else {
        pushFileName(stackPtr, secondElement.value.fileName);
    }
    return 0;
}

// D: rimuove l'elemento in cima. Se è un tensore, decrementa il refcount.
int32_t drop(stack *stackPtr) {
    if (stackPtr->index <= 0) {
        fprintf(stderr, "Errore: drop su stack vuoto\n");
        exit(1);
    }
    stackPtr->index--;
    stackElement topElement = stackPtr->elements[stackPtr->index];
    if (topElement.elementType == TYPE_TENSOR) {
        tensorDeref(topElement.value.tensor);
    } else {
        free(topElement.value.fileName);
    }
    return 0;
}

// 4. Utilità

int32_t getStackSize(stack *stackPtr) {
    return stackPtr->index;
}
