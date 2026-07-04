// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#include "tensor.h"
#include <stdio.h>
#include <immintrin.h> 
#include <omp.h>
#include <stdlib.h>

// OPERAZIONI:
// 1. Operazioni aritmetiche
tensor *sum(tensor *a, tensor *b){
    // 1. Controllo compatibilità dimensioni
    equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
    equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

    // 2. Alloca il tensore del risultato (con la stessa shape di a)
    tensor *resultDiff = allocTensor(a->dimensionOfTensor, a->shape);

    int32_t totalElements = a->shape[0];
    if (a->dimensionOfTensor == 2) {
        totalElements *= a->shape[1];
    }

    float *aData = a->buffer->data;
    float *bData = b->buffer->data;
    float *resultDiffData = resultDiff->buffer->data;

    int32_t limit = totalElements - (totalElements % 8);

    // 3. Regione Parallela OpenMP + AVX
    #pragma omp parallel if(totalElements > 2000)
    {
        // Distribuzione del carico SIMD tra i thread
        #pragma omp for schedule(static) nowait
        for (int32_t i = 0; i < limit; i += 8) {
            __m256 va = _mm256_loadu_ps(&aData[i]);
            __m256 vb = _mm256_loadu_ps(&bData[i]);
            __m256 vres = _mm256_add_ps(va, vb); // <--- Sottrazione nativa AVX (a - b)
            _mm256_storeu_ps(&resultDiffData[i], vres);
        }

        // Il resto viene gestito fuori dal costrutto parallel per evitare race conditions,
        // oppure usando 'single' ma SENZA 'nowait' sopra. 
        // La soluzione più pulita e performante è chiudere il parallel prima o gestire il resto in coda in modo sicuro.
    }

    // 4. Gestione del resto (elementi finali < 8) eseguita in modo sicuro sul thread principale
    for (int32_t i = limit; i < totalElements; i++) {
        resultDiffData[i] = aData[i] - bData[i];
    }

    return resultDiff;
}

tensor *diff(tensor *a, tensor *b){
    // 1. Controllo compatibilità dimensioni
    equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
    equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

    // 2. Alloca il tensore del risultato (con la stessa shape di a)
    tensor *resultDiff = allocTensor(a->dimensionOfTensor, a->shape);

    int32_t totalElements = a->shape[0];
    if (a->dimensionOfTensor == 2) {
        totalElements *= a->shape[1];
    }

    float *aData = a->buffer->data;
    float *bData = b->buffer->data;
    float *resultDiffData = resultDiff->buffer->data;

    int32_t limit = totalElements - (totalElements % 8);

    // 3. Regione Parallela OpenMP + AVX
    #pragma omp parallel if(totalElements > 2000)
    {
        // Distribuzione del carico SIMD tra i thread
        #pragma omp for schedule(static) nowait
        for (int32_t i = 0; i < limit; i += 8) {
            __m256 va = _mm256_loadu_ps(&aData[i]);
            __m256 vb = _mm256_loadu_ps(&bData[i]);
            __m256 vres = _mm256_mul_ps(va, vb); // <--- Sottrazione nativa AVX (a - b)
            _mm256_storeu_ps(&resultDiffData[i], vres);
        }

        // Il resto viene gestito fuori dal costrutto parallel per evitare race conditions,
        // oppure usando 'single' ma SENZA 'nowait' sopra. 
        // La soluzione più pulita e performante è chiudere il parallel prima o gestire il resto in coda in modo sicuro.
    }

    // 4. Gestione del resto (elementi finali < 8) eseguita in modo sicuro sul thread principale
    for (int32_t i = limit; i < totalElements; i++) {
        resultDiffData[i] = aData[i] - bData[i];
    }

    return resultDiff;
}

tensor *prod(tensor *a, tensor *b);

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
tensor *allocTensor(int32_t dimensionOfTensor, const int32_t *shape) {
    // 1. Alloca la struttura logica del tensore
    tensor *t = malloc(sizeof(tensor));
    if (t == NULL) {
        fprintf(stderr, "Impossibile allocare struct tensor");
        exit(1);
    }

    // Inizializza i metadati logici
    t->dimensionOfTensor = dimensionOfTensor;
    t->referenceCount = 1;
    t->shape[0] = shape[0];

    if (dimensionOfTensor > 1) {
        t->shape[1] = shape[1];
    } else {
        t->shape[1] = 1;
    }

    // 2. Alloca la struttura fisica (il buffer condiviso)
    t->buffer = malloc(sizeof(tensor_buffer));
    if (t->buffer == NULL) {
        free(t);
        fprintf(stderr, "Impossibile allocare struct tensor_buffer");
        exit(1);
    }
    
    t->buffer->referenceCount = 1;
    t->buffer->isMMapped = 0;
    t->buffer->mmapAddress = NULL;
    t->buffer->mmapLength = 0;

    // Calcoliamo la dimensione totale del buffer dei dati
    int32_t size = shape[0];
    if (dimensionOfTensor > 1) {
        size *= shape[1];
    }

    // 3. QUI USI posix_memalign per allocare i float del buffer in modo allineato!
    int status = posix_memalign((void**)&(t->buffer->data), 64, size * sizeof(float));
    if (status != 0) {
        free(t->buffer);
        fprintf(stderr, "Impossibile allocare float data");
        exit(1);
    }

    return t;
}

void tensorRef(tensor *t);
void tensorDeref(tensor *t);

// 9. Operazioni di I/O
tensor *readPGM(const char *filename); // (
int writePGM(tensor *t, const char *filename); // )
tensor *readBinary(const char *filename); // {
int writeBinary(tensor *t, const char *filename); // }

// 10. Utilità
void printTensor(tensor *t); // p



// ALTRO
void equalDimensions(int dimensionA, int dimensionB){
    if(dimensionA != dimensionB){
		fprintf(stderr, "Dimensioni non compatibili");
		exit(1);
    } 
}

void equalShapes(int rowA, int rowB, int colA, int colB){
    if((rowA != rowB) || (colA != colB)){
        fprintf(stderr, "Shape non compatibili\n");
		exit(1);
    }
}

