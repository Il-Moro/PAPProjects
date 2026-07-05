// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#include "tensor.h"
#include <stdio.h>
#include <immintrin.h> 
#include <omp.h>
#include <stdlib.h>
#include <sys/mman.h> // per munmap

#define CONSTANT_FOR_PARALLIZATION 200000

// Prototipi helper e costruttori interni per la compilazione
void equalDimensions(int dimensionA, int dimensionB);
void equalShapes(int rowA, int rowB, int colA, int colB);
tensor *allocTensor(int32_t dimensionOfTensor, const int32_t *shape);
int32_t getTotalElements(tensor *t);

// OPERAZIONI:
// 1. Operazioni aritmetiche
tensor *sum(tensor *a, tensor *b){
    // 1. Controllo compatibilità dimensioni
    equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
    equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

    // 2. Alloca il tensore del risultato 
    tensor *resultSum = allocTensor(a->dimensionOfTensor, a->shape);
    int32_t totalElements = getTotalElements(a);

    float *dataOfA = a->buffer->data;
    float *dataOfB = b->buffer->data;
    float *dataResultSum = resultSum->buffer->data;

    int32_t limit = totalElements - (totalElements % 8);

    // 3. Regione Parallela OpenMP + AVX
    #pragma omp parallel if(totalElements > CONSTANT_FOR_PARALLIZATION)
    {
        // Distribuzione del carico SIMD tra i thread
        #pragma omp for schedule(static) nowait
        for (int32_t i = 0; i < limit; i += 8) {
            __m256 va = _mm256_loadu_ps(&dataOfA[i]);
            __m256 vb = _mm256_loadu_ps(&dataOfB[i]);
            __m256 vres = _mm256_add_ps(va, vb); 
            _mm256_storeu_ps(&dataResultSum[i], vres);
        }
    }

    // 4. Gestione del resto (elementi finali < 8) eseguita sul thread principale
    for (int32_t i = limit; i < totalElements; i++) {
        dataResultSum[i] = dataOfA[i] + dataOfB[i];
    }

    return resultSum;
}

tensor *sub(tensor *a, tensor *b){
    // 1. Controllo compatibilità dimensioni
    equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
    equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

    // 2. Alloca il tensore del risultato 
    tensor *resultSum = allocTensor(a->dimensionOfTensor, a->shape);
    int32_t totalElements = getTotalElements(a);

    float *dataOfA = a->buffer->data;
    float *dataOfB = b->buffer->data;
    float *dataResultSum = resultSum->buffer->data;

    int32_t limit = totalElements - (totalElements % 8);

    // 3. Regione Parallela OpenMP + AVX
    #pragma omp parallel if(totalElements > CONSTANT_FOR_PARALLIZATION)
    {
        // Distribuzione del carico SIMD tra i thread
        #pragma omp for schedule(static) nowait
        for (int32_t i = 0; i < limit; i += 8) {
            __m256 va = _mm256_loadu_ps(&dataOfA[i]);
            __m256 vb = _mm256_loadu_ps(&dataOfB[i]);
            __m256 vres = _mm256_sub_ps(va, vb); 
            _mm256_storeu_ps(&dataResultSum[i], vres);
        }
    }
    // 4. Gestione del resto (elementi finali < 8) eseguita sul thread principale
    for (int32_t i = limit; i < totalElements; i++) {
        dataResultSum[i] = dataOfA[i] - dataOfB[i];
    }

    return resultSum;
}

tensor *mul(tensor *a, tensor *b){
    // 1. Controllo compatibilità dimensioni
    equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
    equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

    // 2. Alloca il tensore del risultato 
    tensor *resultSum = allocTensor(a->dimensionOfTensor, a->shape);
    int32_t totalElements = getTotalElements(a);

    float *dataOfA = a->buffer->data;
    float *dataOfB = b->buffer->data;
    float *dataResultSum = resultSum->buffer->data;

    int32_t limit = totalElements - (totalElements % 8);

    // 3. Regione Parallela OpenMP + AVX
    #pragma omp parallel if(totalElements > CONSTANT_FOR_PARALLIZATION)
    {
        // Distribuzione del carico SIMD tra i thread
        #pragma omp for schedule(static) nowait
        for (int32_t i = 0; i < limit; i += 8) {
            __m256 va = _mm256_loadu_ps(&dataOfA[i]);
            __m256 vb = _mm256_loadu_ps(&dataOfB[i]);
            __m256 vres = _mm256_mul_ps(va, vb); // Prodotto nativo AVX (a * b)
            _mm256_storeu_ps(&dataResultSum[i], vres);
        }
    }

    // 4. Gestione del resto (elementi finali < 8) eseguita sul thread principale
    for (int32_t i = limit; i < totalElements; i++) {
        dataResultSum[i] = dataOfA[i] * dataOfB[i];
    }
    return resultSum;
}

// 2. Op. comparazione
tensor *equal(tensor *a, tensor *b){
    // 1. Controllo compatibilità dimensioni
    equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
    equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

    // 2. Alloca il tensore del risultato
    tensor *result = allocTensor(a->dimensionOfTensor, a->shape);
    int32_t totalElements = getTotalElements(a);

    float *dataOfA = a->buffer->data;
    float *dataOfB = b->buffer->data;
    float *dataResult = result->buffer->data;

    int32_t limit = totalElements - (totalElements % 8);

    // Costanti per convertire la maschera di bit in float 1.0 o 0.0
    __m256 vTrueVal = _mm256_set1_ps(1.0f);
    __m256 vFalseVal = _mm256_setzero_ps();

    // 3. Regione Parallela OpenMP + AVX
    #pragma omp parallel if(totalElements > CONSTANT_FOR_PARALLIZATION)
    {
        #pragma omp for schedule(static) nowait
        for (int32_t i = 0; i < limit; i += 8) {
            __m256 va = _mm256_loadu_ps(&dataOfA[i]);
            __m256 vb = _mm256_loadu_ps(&dataOfB[i]);
            
            // Confronta se va == vb elemento per elemento
            __m256 mask = _mm256_cmp_ps(va, vb, _CMP_EQ_OQ); 
            
            // Se il bit della maschera è 1 metti 1.0f, altrimenti 0.0f
            __m256 vres = _mm256_blendv_ps(vFalseVal, vTrueVal, mask);
            
            _mm256_storeu_ps(&dataResult[i], vres);
        }
    }

    // 4. Gestione del resto (elementi finali < 8) sul thread principale
    for (int32_t i = limit; i < totalElements; i++) {
        dataResult[i] = (dataOfA[i] == dataOfB[i]) ? 1.0f : 0.0f;
    }

    return result;
}

tensor *greaterThan(tensor *a, tensor *b){
    equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
    equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

    tensor *result = allocTensor(a->dimensionOfTensor, a->shape);
    int32_t totalElements = getTotalElements(a);

    float *dataOfA = a->buffer->data;
    float *dataOfB = b->buffer->data;
    float *dataResult = result->buffer->data;

    int32_t limit = totalElements - (totalElements % 8);

    // Prepariamo due registri costanti: uno pieno di 1.0f e uno pieno di 0.0f
    __m256 vTrueVal = _mm256_set1_ps(1.0f);
    __m256 vFalseVal = _mm256_setzero_ps(); // setta tutto a 0.0f

    #pragma omp parallel if(totalElements > CONSTANT_FOR_PARALLIZATION)
    {
        #pragma omp for schedule(static) nowait
        for (int32_t i = 0; i < limit; i += 8) {
            __m256 va = _mm256_loadu_ps(&dataOfA[i]);
            __m256 vb = _mm256_loadu_ps(&dataOfB[i]);
            
            // 1. Esegui la comparazione (ritorna la maschera di bit)
            __m256 mask = _mm256_cmp_ps(va, vb, _CMP_GT_OQ); 
            
            // 2. Converti la maschera in 1.0f o 0.0f usando il blend
            __m256 vres = _mm256_blendv_ps(vFalseVal, vTrueVal, mask);
            
            _mm256_storeu_ps(&dataResult[i], vres);
        }
    }

    // 4. Gestione del resto (elementi finali < 8)
    for (int32_t i = limit; i < totalElements; i++) {
        // Nel C standard la comparazione restituisce 1 o 0, 
        // assegnandolo al float diventa automaticamente 1.0f o 0.0f
        dataResult[i] = (dataOfA[i] > dataOfB[i]) ? 1.0f : 0.0f;
    }
    return result;
}

tensor *lessThan(tensor *a, tensor *b){
    // 1. Controllo compatibilità dimensioni
    equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
    equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

    // 2. Alloca il tensore del risultato
    tensor *result = allocTensor(a->dimensionOfTensor, a->shape);
    int32_t totalElements = getTotalElements(a);

    float *dataOfA = a->buffer->data;
    float *dataOfB = b->buffer->data;
    float *dataResult = result->buffer->data;

    int32_t limit = totalElements - (totalElements % 8);

    // Costanti per convertire la maschera di bit in float 1.0f o 0.0f
    __m256 vTrueVal = _mm256_set1_ps(1.0f);
    __m256 vFalseVal = _mm256_setzero_ps();

    // 3. Regione Parallela OpenMP + AVX
    #pragma omp parallel if(totalElements > CONSTANT_FOR_PARALLIZATION)
    {
        #pragma omp for schedule(static) nowait
        for (int32_t i = 0; i < limit; i += 8) {
            __m256 va = _mm256_loadu_ps(&dataOfA[i]);
            __m256 vb = _mm256_loadu_ps(&dataOfB[i]);
            
            // Confronta se va < vb elemento per elemento
            __m256 mask = _mm256_cmp_ps(va, vb, _CMP_LT_OQ); 
            
            // Se la condizione è vera mette 1.0f, altrimenti 0.0f
            __m256 vres = _mm256_blendv_ps(vFalseVal, vTrueVal, mask);
            
            _mm256_storeu_ps(&dataResult[i], vres);
        }
    }

    // 4. Gestione del resto (elementi finali < 8) sul thread principale
    for (int32_t i = limit; i < totalElements; i++) {
        dataResult[i] = (dataOfA[i] < dataOfB[i]) ? 1.0f : 0.0f;
    }

    return result;
}


// 3. Op. Logiche
tensor *AND(tensor *a, tensor *b){
    equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
    equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

    tensor *result = allocTensor(a->dimensionOfTensor, a->shape);
    int32_t totalElements = getTotalElements(a);

    float *dataOfA = a->buffer->data;
    float *dataOfB = b->buffer->data;
    float *dataResult = result->buffer->data;

    int32_t limit = totalElements - (totalElements % 8);

    __m256 vTrueVal = _mm256_set1_ps(1.0f);
    __m256 vFalseVal = _mm256_setzero_ps();

    #pragma omp parallel if(totalElements > CONSTANT_FOR_PARALLIZATION)
    {
        #pragma omp for schedule(static) nowait
        for (int32_t i = 0; i < limit; i += 8) {
            __m256 va = _mm256_loadu_ps(&dataOfA[i]);
            __m256 vb = _mm256_loadu_ps(&dataOfB[i]);
            
            // Crea le maschere di bit dove il valore è pari a 1.0f
            __m256 maskA = _mm256_cmp_ps(va, vTrueVal, _CMP_EQ_OQ);
            __m256 maskB = _mm256_cmp_ps(vb, vTrueVal, _CMP_EQ_OQ);
            
            // Esegue l'AND logico bit a bit tra le due maschere
            __m256 maskRes = _mm256_and_ps(maskA, maskB);
            
            // Converte il risultato binario in float 1.0f o 0.0f
            __m256 vres = _mm256_blendv_ps(vFalseVal, vTrueVal, maskRes);
            
            _mm256_storeu_ps(&dataResult[i], vres);
        }
    }

    for (int32_t i = limit; i < totalElements; i++) {
        dataResult[i] = ((dataOfA[i] == 1.0f) && (dataOfB[i] == 1.0f)) ? 1.0f : 0.0f;
    }

    return result;
}

tensor *OR(tensor *a, tensor *b){
    equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
    equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);

    tensor *result = allocTensor(a->dimensionOfTensor, a->shape);
    int32_t totalElements = getTotalElements(a);

    float *dataOfA = a->buffer->data;
    float *dataOfB = b->buffer->data;
    float *dataResult = result->buffer->data;

    int32_t limit = totalElements - (totalElements % 8);

    __m256 vTrueVal = _mm256_set1_ps(1.0f);
    __m256 vFalseVal = _mm256_setzero_ps();

    #pragma omp parallel if(totalElements > CONSTANT_FOR_PARALLIZATION)
    {
        #pragma omp for schedule(static) nowait
        for (int32_t i = 0; i < limit; i += 8) {
            __m256 va = _mm256_loadu_ps(&dataOfA[i]);
            __m256 vb = _mm256_loadu_ps(&dataOfB[i]);
            
            __m256 maskA = _mm256_cmp_ps(va, vTrueVal, _CMP_EQ_OQ);
            __m256 maskB = _mm256_cmp_ps(vb, vTrueVal, _CMP_EQ_OQ);
            
            // Esegue l'OR logico bit a bit
            __m256 maskRes = _mm256_or_ps(maskA, maskB);
            
            __m256 vres = _mm256_blendv_ps(vFalseVal, vTrueVal, maskRes);
            
            _mm256_storeu_ps(&dataResult[i], vres);
        }
    }

    for (int32_t i = limit; i < totalElements; i++) {
        dataResult[i] = ((dataOfA[i] == 1.0f) || (dataOfB[i] == 1.0f)) ? 1.0f : 0.0f;
    }

    return result;
}

tensor *NOT(tensor *a){
    // Alloca il risultato speculare ad a
    tensor *result = allocTensor(a->dimensionOfTensor, a->shape);
    int32_t totalElements = getTotalElements(a); 

    float *dataOfA = a->buffer->data;
    float *dataResult = result->buffer->data;

    int32_t limit = totalElements - (totalElements % 8);

    __m256 vTrueVal = _mm256_set1_ps(1.0f);
    __m256 vFalseVal = _mm256_setzero_ps();

    #pragma omp parallel if(totalElements > CONSTANT_FOR_PARALLIZATION)
    {
        #pragma omp for schedule(static) nowait
        for (int32_t i = 0; i < limit; i += 8) {
            __m256 va = _mm256_loadu_ps(&dataOfA[i]);
            
            // Trova dove va == 1.0f (restituisce tutti 1 binari se vero)
            __m256 maskA = _mm256_cmp_ps(va, vTrueVal, _CMP_EQ_OQ);
            
            // Invertiamo la logica del blend: 
            // Se la maschera dice VERO (era 1.0f), inseriamo vFalseVal (0.0f)
            // Se la maschera dice FALSO, inseriamo vTrueVal (1.0f)
            __m256 vres = _mm256_blendv_ps(vTrueVal, vFalseVal, maskA);
            
            _mm256_storeu_ps(&dataResult[i], vres);
        }
    }

    for (int32_t i = limit; i < totalElements; i++) {
        dataResult[i] = (dataOfA[i] == 1.0f) ? 0.0f : 1.0f;
    }

    return result;
}

// 4. Selezione
tensor *selection(tensor *a, tensor *b, tensor *mask){
    // 1. Controllo compatibilità dimensioni tra tutti e tre i tensori
    equalDimensions(a->dimensionOfTensor, b->dimensionOfTensor);
    equalShapes(a->shape[0], b->shape[0], a->shape[1], b->shape[1]);
    
    equalDimensions(a->dimensionOfTensor, mask->dimensionOfTensor);
    equalShapes(a->shape[0], mask->shape[0], a->shape[1], mask->shape[1]);

    // 2. Alloca il tensore del risultato
    tensor *result = allocTensor(a->dimensionOfTensor, a->shape);

    int32_t totalElements = getTotalElements(a);

    float *dataOfA = a->buffer->data;
    float *dataOfB = b->buffer->data;
    float *dataMask = mask->buffer->data;
    float *dataResult = result->buffer->data;

    int32_t limit = totalElements - (totalElements % 8);

    // Registro costante con 1.0f per valutare la maschera inserita dall'utente
    __m256 vTrueVal = _mm256_set1_ps(1.0f);

    // 3. Regione Parallela OpenMP + AVX
    #pragma omp parallel if(totalElements > CONSTANT_FOR_PARALLIZATION)
    {
        #pragma omp for schedule(static) nowait
        for (int32_t i = 0; i < limit; i += 8) {
            __m256 va = _mm256_loadu_ps(&dataOfA[i]);
            __m256 vb = _mm256_loadu_ps(&dataOfB[i]);
            __m256 vmask = _mm256_loadu_ps(&dataMask[i]);
            
            // Generiamo la vera maschera di bit hardware: 0xFFFFFFFF dove vmask == 1.0f
            __m256 bitMask = _mm256_cmp_ps(vmask, vTrueVal, _CMP_EQ_OQ);
            
            // Scegliamo: se il bit della maschera è 1 prendiamo da va (VERO), altrimenti vb (FALSO)
            __m256 vres = _mm256_blendv_ps(vb, va, bitMask);
            
            _mm256_storeu_ps(&dataResult[i], vres);
        }
    }

    // 4. Gestione del resto (elementi finali < 8) sul thread principale
    for (int32_t i = limit; i < totalElements; i++) {
        // Se mask[i] è 1.0f prendi a[i], altrimenti b[i]
        dataResult[i] = (dataMask[i] == 1.0f) ? dataOfA[i] : dataOfB[i];
    }

    return result;
}

// 5. Op. specifiche per tensori
tensor *matrixMul(tensor *a, tensor *b){
    equalDimensions(a->dimensionOfTensor, 2);
    equalDimensions(b->dimensionOfTensor, 2);
    equalShapes(a->shape[1], b->shape[0], 0, 0);

}
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

void tensorRef(tensor *t) {
    if (t != NULL) {
        t->referenceCount++;
    }
}

void tensorDeref(tensor *t) {
    if (t == NULL) return;
    t->referenceCount--;
    if (t->referenceCount == 0) {
        // Se il tensore logico non è più referenziato, decrementiamo il buffer fisico
        if (t->buffer != NULL) {
            t->buffer->referenceCount--;
            if (t->buffer->referenceCount == 0) {
                // Se il buffer non ha più proprietari logici, lo liberiamo
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
		fprintf(stderr, "Dimensioni non compati bili");
		exit(1);
    } 
}

void equalShapes(int a, int b, int c, int d){
    if((a != b) || (c != d)){
        fprintf(stderr, "Shape non compatibili\n");
		exit(1);
    }
}

int32_t getTotalElements(tensor *t) {
    return (t->dimensionOfTensor == 2) ? (t->shape[0] * t->shape[1]) : t->shape[0];
}

