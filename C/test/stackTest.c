// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#include "stack.h"
#include "tensor.h"

// Macro per testare che un'operazione su stack vuoto termini con exit(1)
#define TEST_EXITS(body) do {                              \
    pid_t _pid = fork();                                   \
    assert(_pid >= 0);                                     \
    if (_pid == 0) {                                       \
        freopen("/dev/null", "w", stderr);                 \
        body;                                              \
        exit(0);                                           \
    } else {                                               \
        int _st; waitpid(_pid, &_st, 0);                  \
        assert(WIFEXITED(_st) && WEXITSTATUS(_st) == 1);  \
    }                                                      \
} while(0)

// --------------------------------------------------------------------------
// 1. Inizializzazione e dimensione
// --------------------------------------------------------------------------
void testStackInit() {
    printf("[TEST] testStackInit...\n");
    stack *stackPtr = stackInit(4);
    assert(stackPtr != NULL);
    assert(getStackSize(stackPtr) == 0);
    stackDestroy(stackPtr);
    printf("[TEST] testStackInit superato!\n");
}

// --------------------------------------------------------------------------
// 2. Push e pop di tensori con reference counting
// --------------------------------------------------------------------------
void testPushPopTensor() {
    printf("[TEST] testPushPopTensor...\n");
    int32_t shape[] = {3};
    tensor *tensorT = allocTensor(1, shape);
    tensorT->buffer->data[0] = 1.0f;
    tensorT->buffer->data[1] = 2.0f;
    tensorT->buffer->data[2] = 3.0f;

    stack *stackPtr = stackInit(4);

    // pushTensor deve incrementare il refcount
    pushTensor(stackPtr, tensorT);
    assert(tensorT->referenceCount == 2); // 1 (originale) + 1 (stack)
    assert(getStackSize(stackPtr) == 1);

    stackElement poppedElement = pop(stackPtr);
    assert(poppedElement.elementType == TYPE_TENSOR);
    assert(poppedElement.value.tensor == tensorT);
    // pop non decrementa: il chiamante è ora owner
    assert(tensorT->referenceCount == 2);
    assert(getStackSize(stackPtr) == 0);

    tensorDeref(tensorT); // rilascia il riferimento del pop
    tensorDeref(tensorT); // rilascia il riferimento originale
    stackDestroy(stackPtr);
    printf("[TEST] testPushPopTensor superato!\n");
}

// --------------------------------------------------------------------------
// 3. Push e pop di stringhe
// --------------------------------------------------------------------------
void testPushPopString() {
    printf("[TEST] testPushPopString...\n");
    stack *stackPtr = stackInit(4);

    pushFileName(stackPtr, "immagine.pgm");
    assert(getStackSize(stackPtr) == 1);

    stackElement poppedElement = pop(stackPtr);
    assert(poppedElement.elementType == TYPE_STRING);
    assert(strcmp(poppedElement.value.fileName, "immagine.pgm") == 0);
    free(poppedElement.value.fileName); // il chiamante è responsabile

    assert(getStackSize(stackPtr) == 0);
    stackDestroy(stackPtr);
    printf("[TEST] testPushPopString superato!\n");
}

// --------------------------------------------------------------------------
// 4. peek: ispeziona senza rimuovere
// --------------------------------------------------------------------------
void testPeek() {
    printf("[TEST] testPeek...\n");
    int32_t shape[] = {2};
    tensor *tensorA = allocTensor(1, shape);
    tensor *tensorB = allocTensor(1, shape);
    tensorA->buffer->data[0] = 10.0f;
    tensorB->buffer->data[0] = 20.0f;

    stack *stackPtr = stackInit(4);
    pushTensor(stackPtr, tensorA);
    pushTensor(stackPtr, tensorB);

    // peek(0) = cima = tensorB, peek(1) = sotto = tensorA
    stackElement top   = peek(stackPtr, 0);
    stackElement below = peek(stackPtr, 1);
    assert(top.value.tensor == tensorB);
    assert(below.value.tensor == tensorA);
    assert(getStackSize(stackPtr) == 2); // invariato

    tensorDeref(tensorA);
    tensorDeref(tensorB);
    stackDestroy(stackPtr); // destroy chiama tensorDeref internamente
    printf("[TEST] testPeek superato!\n");
}

// --------------------------------------------------------------------------
// 5. stackDup: duplica la cima, incrementa refcount
// --------------------------------------------------------------------------
void testDup() {
    printf("[TEST] testDup...\n");
    int32_t shape[] = {2};
    tensor *tensorT = allocTensor(1, shape);
    tensorT->buffer->data[0] = 7.0f;

    stack *stackPtr = stackInit(4);
    pushTensor(stackPtr, tensorT);
    assert(tensorT->referenceCount == 2);

    int returnCode = stackDup(stackPtr);
    assert(returnCode == 0);
    assert(getStackSize(stackPtr) == 2);
    // stackDup ha chiamato pushTensor di nuovo → refcount sale a 3
    assert(tensorT->referenceCount == 3);

    // I due elementi in cima puntano allo stesso tensore
    stackElement top1 = pop(stackPtr);
    stackElement top2 = pop(stackPtr);
    assert(top1.value.tensor == tensorT);
    assert(top2.value.tensor == tensorT);

    tensorDeref(tensorT); // deref pop 1
    tensorDeref(tensorT); // deref pop 2
    tensorDeref(tensorT); // deref originale
    stackDestroy(stackPtr);
    printf("[TEST] testDup superato!\n");
}

// --------------------------------------------------------------------------
// 6. swap: scambia i due elementi in cima
// --------------------------------------------------------------------------
void testSwap() {
    printf("[TEST] testSwap...\n");
    int32_t shape[] = {1};
    tensor *tensorA = allocTensor(1, shape); tensorA->buffer->data[0] = 1.0f;
    tensor *tensorB = allocTensor(1, shape); tensorB->buffer->data[0] = 2.0f;

    stack *stackPtr = stackInit(4);
    pushTensor(stackPtr, tensorA); // fondo
    pushTensor(stackPtr, tensorB); // cima

    int returnCode = swap(stackPtr);
    assert(returnCode == 0);

    // Dopo swap: tensorA in cima, tensorB in fondo
    stackElement top = pop(stackPtr);
    assert(top.value.tensor->buffer->data[0] == 1.0f); // tensorA
    stackElement bot = pop(stackPtr);
    assert(bot.value.tensor->buffer->data[0] == 2.0f); // tensorB

    // swap non modifica refcount → liberiamo manualmente
    tensorDeref(tensorA); tensorDeref(tensorA); // 1 push + 1 pop
    tensorDeref(tensorB); tensorDeref(tensorB);
    stackDestroy(stackPtr);
    printf("[TEST] testSwap superato!\n");
}

// --------------------------------------------------------------------------
// 7. over: copia il secondo elemento in cima
// --------------------------------------------------------------------------
void testOver() {
    printf("[TEST] testOver...\n");
    int32_t shape[] = {1};
    tensor *tensorA = allocTensor(1, shape); tensorA->buffer->data[0] = 5.0f;
    tensor *tensorB = allocTensor(1, shape); tensorB->buffer->data[0] = 9.0f;

    stack *stackPtr = stackInit(4);
    pushTensor(stackPtr, tensorA); // fondo: tensorA
    pushTensor(stackPtr, tensorB); // cima:  tensorB

    int returnCode = over(stackPtr);
    assert(returnCode == 0);
    assert(getStackSize(stackPtr) == 3);
    // Stack: [tensorA, tensorB, tensorACopy] con tensorACopy == tensorA (stesso puntatore)
    assert(tensorA->referenceCount == 3); // push + push_over + originale

    stackElement top = pop(stackPtr);
    assert(top.value.tensor == tensorA);

    tensorDeref(tensorA); tensorDeref(tensorB);
    stackDestroy(stackPtr);
    printf("[TEST] testOver superato!\n");
}

// --------------------------------------------------------------------------
// 8. drop: rimuove la cima e decrementa refcount
// --------------------------------------------------------------------------
void testDrop() {
    printf("[TEST] testDrop...\n");
    int32_t shape[] = {1};
    tensor *tensorT = allocTensor(1, shape);

    stack *stackPtr = stackInit(4);
    pushTensor(stackPtr, tensorT);
    assert(tensorT->referenceCount == 2);

    int returnCode = drop(stackPtr);
    assert(returnCode == 0);
    assert(getStackSize(stackPtr) == 0);
    // drop ha chiamato tensorDeref → refcount torna a 1
    assert(tensorT->referenceCount == 1);

    tensorDeref(tensorT); // rilascia originale
    stackDestroy(stackPtr);
    printf("[TEST] testDrop superato!\n");
}

// --------------------------------------------------------------------------
// 9. crescita dinamica: push oltre la capacità iniziale
// --------------------------------------------------------------------------
void testDynamicGrowth() {
    printf("[TEST] testDynamicGrowth...\n");
    stack *stackPtr = stackInit(2); // capacità iniziale piccola

    int32_t shape[] = {1};
    tensor *tensors[20];
    for (int i = 0; i < 20; i++) {
        tensors[i] = allocTensor(1, shape);
        tensors[i]->buffer->data[0] = (float)i;
        pushTensor(stackPtr, tensors[i]);
    }
    assert(getStackSize(stackPtr) == 20);

    for (int i = 19; i >= 0; i--) {
        stackElement poppedElement = pop(stackPtr);
        assert(poppedElement.value.tensor->buffer->data[0] == (float)i);
        tensorDeref(tensors[i]); // deref pop
        tensorDeref(tensors[i]); // deref originale
    }
    stackDestroy(stackPtr);
    printf("[TEST] testDynamicGrowth superato!\n");
}

// --------------------------------------------------------------------------
// 10. Errori su stack vuoto → exit(1)
// --------------------------------------------------------------------------
void testErrorsOnEmptyStack() {
    printf("[TEST] testErrorsOnEmptyStack...\n");

    TEST_EXITS({ stack *stackPtr = stackInit(4); pop(stackPtr); });
    printf("  pop su stack vuoto: OK\n");

    TEST_EXITS({ stack *stackPtr = stackInit(4); stackDup(stackPtr); });
    printf("  stackDup su stack vuoto: OK\n");

    TEST_EXITS({ stack *stackPtr = stackInit(4); drop(stackPtr); });
    printf("  drop su stack vuoto: OK\n");

    TEST_EXITS({
        stack *stackPtr = stackInit(4);
        int32_t sh[] = {1};
        tensor *tensorT = allocTensor(1, sh);
        pushTensor(stackPtr, tensorT);
        swap(stackPtr); // solo 1 elemento: deve fallire
    });
    printf("  swap con 1 elemento: OK\n");

    TEST_EXITS({
        stack *stackPtr = stackInit(4);
        int32_t sh[] = {1};
        tensor *tensorT = allocTensor(1, sh);
        pushTensor(stackPtr, tensorT);
        over(stackPtr); // solo 1 elemento: deve fallire
    });
    printf("  over con 1 elemento: OK\n");

    printf("[TEST] testErrorsOnEmptyStack superato!\n");
}

// --------------------------------------------------------------------------
// 11. stackDestroy libera i tensori rimasti
// --------------------------------------------------------------------------
void testDestroyFreesRemaining() {
    printf("[TEST] testDestroyFreesRemaining...\n");
    int32_t shape[] = {4};
    tensor *tensorT = allocTensor(1, shape);

    stack *stackPtr = stackInit(4);
    pushTensor(stackPtr, tensorT); // refcount = 2
    pushTensor(stackPtr, tensorT); // refcount = 3

    // destroy deve fare tensorDeref x2 → refcount torna a 1
    stackDestroy(stackPtr);
    assert(tensorT->referenceCount == 1);
    tensorDeref(tensorT); // libera l'originale
    printf("[TEST] testDestroyFreesRemaining superato!\n");
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------
int main() {
    printf("=== AVVIO SUITE COMPLETA TEST STACK ===\n");
    testStackInit();
    testPushPopTensor();
    testPushPopString();
    testPeek();
    testDup();
    testSwap();
    testOver();
    testDrop();
    testDynamicGrowth();
    testErrorsOnEmptyStack();
    testDestroyFreesRemaining();
    printf("=== TUTTI I TEST STACK PASSATI CON SUCCESSO! ===\n");
    return 0;
}
