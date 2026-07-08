// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#include <stdio.h>
#include <stdlib.h>

#include "stack/stack.h"
#include "tensor/tensor.h"
#include "parsing/token.h"

// Funzioni di supporto per pop con type-checking

// Estrae il tensore in cima allo stack.
// Se la cima non è un tensore, segnala errore ed esce.
// Il chiamante diventa owner del tensore (responsabile di tensorDeref).
static tensor *popTensorChecked(stack *stackPtr, const char *opName) {
    if (getStackSize(stackPtr) == 0) {
        fprintf(stderr, "Errore '%s': stack vuoto\n", opName);
        exit(1);
    }
    stackElement el = pop(stackPtr);
    if (el.elementType != TYPE_TENSOR) {
        fprintf(stderr, "Errore '%s': atteso tensore, trovato stringa\n", opName);
        exit(1);
    }
    return el.value.tensor;
}

// Estrae la stringa in cima allo stack.
// Se la cima non è una stringa, segnala errore ed esce.
// Il chiamante diventa owner della stringa (responsabile di free).
static char *popStringChecked(stack *stackPtr, const char *opName) {
    if (getStackSize(stackPtr) == 0) {
        fprintf(stderr, "Errore '%s': stack vuoto\n", opName);
        exit(1);
    }
    stackElement el = pop(stackPtr);
    if (el.elementType != TYPE_STRING) {
        fprintf(stderr, "Errore '%s': atteso filename (stringa), trovato tensore\n", opName);
        exit(1);
    }
    return el.value.fileName;
}

// Ciclo interprete: esegue i token uno alla volta sullo stack

// Esegue un array di Token su uno stack TensorForth.
// Per ogni operazione binaria (b a -- res): 'a' è in cima, 'b' è sotto.
// Dopo l'esecuzione, i tensori intermedi vengono derefenziati.
static void execute(Token *tokenArray, size_t tokenCount, stack *stackPtr) {
    for (size_t tokenIndex = 0; tokenIndex < tokenCount; tokenIndex++) {
        Token currentToken = tokenArray[tokenIndex];

        switch (currentToken.type) {

        // Letterali: push sullo stack
        case TOKEN_TENSOR_LITERAL:
            // pushTensor incrementa refcount
            pushTensor(stackPtr, currentToken.value.t);
            break;

        case TOKEN_STRING:
            pushFileName(stackPtr, currentToken.value.str);
            break;

        // Operazioni aritmetiche  ( b a -- res )
        case TOKEN_ADD: {
            tensor *tensorA = popTensorChecked(stackPtr, "+");
            tensor *tensorB = popTensorChecked(stackPtr, "+");
            tensor *resultTensor = sum(tensorA, tensorB);
            tensorDeref(tensorA);
            tensorDeref(tensorB);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor); // pushTensor ha già incrementato
            break;
        }
        case TOKEN_SUB: {
            tensor *tensorA = popTensorChecked(stackPtr, "-");
            tensor *tensorB = popTensorChecked(stackPtr, "-");
            tensor *resultTensor = sub(tensorA, tensorB);   // a - b
            tensorDeref(tensorA);
            tensorDeref(tensorB);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }
        case TOKEN_MUL: {
            tensor *tensorA = popTensorChecked(stackPtr, "*");
            tensor *tensorB = popTensorChecked(stackPtr, "*");
            tensor *resultTensor = mul(tensorA, tensorB);
            tensorDeref(tensorA);
            tensorDeref(tensorB);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }

        // Operazioni di comparazione  ( b a -- res )
        case TOKEN_LT: {
            tensor *tensorA = popTensorChecked(stackPtr, "<");
            tensor *tensorB = popTensorChecked(stackPtr, "<");
            tensor *resultTensor = lessThan(tensorA, tensorB);   // a < b
            tensorDeref(tensorA);
            tensorDeref(tensorB);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }
        case TOKEN_GT: {
            tensor *tensorA = popTensorChecked(stackPtr, ">");
            tensor *tensorB = popTensorChecked(stackPtr, ">");
            tensor *resultTensor = greaterThan(tensorA, tensorB); // a > b
            tensorDeref(tensorA);
            tensorDeref(tensorB);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }
        case TOKEN_EQ: {
            tensor *tensorA = popTensorChecked(stackPtr, "=");
            tensor *tensorB = popTensorChecked(stackPtr, "=");
            tensor *resultTensor = equal(tensorA, tensorB);
            tensorDeref(tensorA);
            tensorDeref(tensorB);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }

        // Operazioni logiche
        case TOKEN_AND: {
            tensor *tensorA = popTensorChecked(stackPtr, "&");
            tensor *tensorB = popTensorChecked(stackPtr, "&");
            tensor *resultTensor = AND(tensorA, tensorB);
            tensorDeref(tensorA);
            tensorDeref(tensorB);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }
        case TOKEN_OR: {
            tensor *tensorA = popTensorChecked(stackPtr, "|");
            tensor *tensorB = popTensorChecked(stackPtr, "|");
            tensor *resultTensor = OR(tensorA, tensorB);
            tensorDeref(tensorA);
            tensorDeref(tensorB);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }
        case TOKEN_NOT: {
            tensor *tensorA = popTensorChecked(stackPtr, "!");
            tensor *resultTensor = NOT(tensorA);
            tensorDeref(tensorA);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }

        // Selezione  ( b a m -- res )
        case TOKEN_SELECT: {
            tensor *maskTensor = popTensorChecked(stackPtr, "$");  // maschera in cima
            tensor *tensorA = popTensorChecked(stackPtr, "$");  // valore se 1
            tensor *tensorB = popTensorChecked(stackPtr, "$");  // valore se 0
            tensor *resultTensor = selection(tensorA, tensorB, maskTensor);
            tensorDeref(maskTensor);
            tensorDeref(tensorA);
            tensorDeref(tensorB);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }

        // Operatori su tensori
        case TOKEN_MATMUL: {
            tensor *tensorA = popTensorChecked(stackPtr, "@");  // in cima
            tensor *tensorB = popTensorChecked(stackPtr, "@");  // sotto
            tensor *resultTensor = matrixMul(tensorA, tensorB);         // a @ b
            tensorDeref(tensorA);
            tensorDeref(tensorB);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }
        case TOKEN_DOT: {
            tensor *tensorA = popTensorChecked(stackPtr, ".");
            tensor *tensorB = popTensorChecked(stackPtr, ".");
            tensor *resultTensor = dotProduct(tensorA, tensorB);
            tensorDeref(tensorA);
            tensorDeref(tensorB);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }
        case TOKEN_CONV2D: {
            tensor *kernelTensor = popTensorChecked(stackPtr, "c");  // kernel in cima
            tensor *tensorA = popTensorChecked(stackPtr, "c");  // immagine sotto
            tensor *resultTensor = conv2D(tensorA, kernelTensor);
            tensorDeref(kernelTensor);
            tensorDeref(tensorA);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }

        // Forma e dimensione
        case TOKEN_RESHAPE: {
            tensor *shapeTensor = popTensorChecked(stackPtr, "r");  // shape in cima
            tensor *tensorA     = popTensorChecked(stackPtr, "r");  // tensore sotto
            tensor *resultTensor   = reshape(tensorA, shapeTensor);
            tensorDeref(shapeTensor);
            tensorDeref(tensorA);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }
        case TOKEN_RAVEL: {
            tensor *tensorA   = popTensorChecked(stackPtr, "_");
            tensor *resultTensor = ravel(tensorA);
            tensorDeref(tensorA);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }
        case TOKEN_SHAPE: {
            tensor *tensorA   = popTensorChecked(stackPtr, "#");
            tensor *resultTensor = getShape(tensorA);
            tensorDeref(tensorA);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }

        // Generazione, riduzione, riempimento
        case TOKEN_RANDOM: {
            tensor *shapeTensor = popTensorChecked(stackPtr, "?");
            tensor *resultTensor   = randomTensor(shapeTensor);
            tensorDeref(shapeTensor);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }
        case TOKEN_RELU: {
            tensor *tensorA   = popTensorChecked(stackPtr, "R");
            tensor *resultTensor = relu(tensorA);
            tensorDeref(tensorA);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }
        case TOKEN_MIN: {
            tensor *tensorA   = popTensorChecked(stackPtr, "m");
            tensor *tensorB   = popTensorChecked(stackPtr, "m");
            tensor *resultTensor = minTensor(tensorA, tensorB);
            tensorDeref(tensorA);
            tensorDeref(tensorB);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }
        case TOKEN_MAX: {
            tensor *tensorA   = popTensorChecked(stackPtr, "M");
            tensor *tensorB   = popTensorChecked(stackPtr, "M");
            tensor *resultTensor = maxTensor(tensorA, tensorB);
            tensorDeref(tensorA);
            tensorDeref(tensorB);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }
        case TOKEN_REDUCTION_SUM: {
            tensor *tensorA   = popTensorChecked(stackPtr, "S");
            tensor *resultTensor = reductionSum(tensorA);
            tensorDeref(tensorA);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }
        case TOKEN_FILL: {
            tensor *valTensor   = popTensorChecked(stackPtr, "f");  // valori in cima
            tensor *shapeTensor = popTensorChecked(stackPtr, "f");  // shape sotto
            tensor *resultTensor   = fill(shapeTensor, valTensor);
            tensorDeref(valTensor);
            tensorDeref(shapeTensor);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }

        // I/O
        case TOKEN_READ_PGM: {
            char *fileName = popStringChecked(stackPtr, "(");
            tensor *resultTensor = readPGM(fileName);
            free(fileName);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }
        case TOKEN_WRITE_PGM: {
            char   *fileName = popStringChecked(stackPtr, ")");
            tensor *tensorToWrite = popTensorChecked(stackPtr, ")");
            writePGM(tensorToWrite, fileName);
            tensorDeref(tensorToWrite);
            free(fileName);
            break;
        }
        case TOKEN_READ_BINARY: {
            char *fileName = popStringChecked(stackPtr, "{");
            tensor *resultTensor = readBinary(fileName);
            free(fileName);
            pushTensor(stackPtr, resultTensor);
            tensorDeref(resultTensor);
            break;
        }
        case TOKEN_WRITE_BINARY: {
            char   *fileName = popStringChecked(stackPtr, "}");
            tensor *tensorToWrite = popTensorChecked(stackPtr, "}");
            writeBinary(tensorToWrite, fileName);
            tensorDeref(tensorToWrite);
            free(fileName);
            break;
        }

        // Manipolazione stack
        case TOKEN_DUP:  stackDup(stackPtr);  break;
        case TOKEN_SWAP: swap(stackPtr); break;
        case TOKEN_OVER: over(stackPtr); break;
        case TOKEN_DROP: drop(stackPtr); break;

        // Utilità: p consuma il tensore (come da specifica)
        case TOKEN_PRINT: {
            tensor *tensorA = popTensorChecked(stackPtr, "p");
            printTensor(tensorA);
            tensorDeref(tensorA);
            break;
        }

        case TOKEN_UNKNOWN:
        default:
            fprintf(stderr, "Errore: token sconosciuto durante l'esecuzione\n");
            exit(1);
        }
    }
}

// main

int main(int argumentCount, char *argumentValues[]) {
    if (argumentCount < 2) {
        fprintf(stderr, "Errore: file sorgente richiesto in input\n");
        fprintf(stderr, "Uso: tensorforth <file.tf>\n");
        return 1;
    }

    // 1. Tokenizza il file sorgente
    size_t  tokenCount = 0;
    Token  *tokenArray = tokenize(argumentValues[1], &tokenCount);

    // 2. Crea lo stack
    stack *stackPtr = stackInit(32);

    // 3. Esegui i token
    execute(tokenArray, tokenCount, stackPtr);

    // 4. Pulizia
    stackDestroy(stackPtr);
    freeTokens(tokenArray, tokenCount);

    return 0;
}
