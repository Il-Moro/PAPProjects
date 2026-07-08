// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>

#include "tensor.h"

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

// 1. Scrittura e lettura PGM
void test_pgm_io() {
    printf("[TEST IO] test_pgm_io...\n");

    int32_t shape[] = {4, 4};
    tensor *t = allocTensor(2, shape);
    // Riempiamo il tensore con un pattern (gradiente)
    float *data = t->buffer->data;
    for (int i = 0; i < 16; i++) {
        data[i] = (float)i / 15.0f;
    }

    // Scrive su file PGM
    const char *filename = "/tmp/io_test_image.pgm";
    int success = writePGM(t, filename);
    assert(success == 1);

    // Legge dal file PGM
    tensor *read_t = readPGM(filename);
    assert(read_t != NULL);
    assert(read_t->dimensionOfTensor == 2);
    assert(read_t->shape[0] == 4);
    assert(read_t->shape[1] == 4);

    // Verifica i pixel normalizzati (differenza < epsilon a causa della discretizzazione a 8 bit)
    float *read_data = read_t->buffer->data;
    for (int i = 0; i < 16; i++) {
        float expected = (float)i / 15.0f;
        float actual = read_data[i];
        assert(fabs(actual - expected) < 1.0f / 254.0f);
    }

    tensorDeref(t);
    tensorDeref(read_t);
    unlink(filename);

    printf("[TEST IO] test_pgm_io superato!\n");
}

// 2. Scrittura e lettura file binario (con mmap e allineamento)
void test_binary_io() {
    printf("[TEST IO] test_binary_io...\n");

    int32_t shape[] = {5, 3};
    tensor *t = allocTensor(2, shape);
    float *data = t->buffer->data;
    for (int i = 0; i < 15; i++) {
        data[i] = (float)i * 1.5f - 3.2f;
    }

    // Scrive in formato binario
    const char *filename = "/tmp/io_test_binary.bin";
    int success = writeBinary(t, filename);
    assert(success == 1);

    // Legge in mmap
    tensor *read_t = readBinary(filename);
    assert(read_t != NULL);
    assert(read_t->dimensionOfTensor == 2);
    assert(read_t->shape[0] == 5);
    assert(read_t->shape[1] == 3);
    assert(read_t->buffer->isMMapped == 1);
    assert(read_t->buffer->mmapAddress != NULL);
    assert(read_t->buffer->mmapLength > 64);

    // Controlliamo che l'allineamento a 64 byte sia rispettato per l'indirizzo dei dati
    uintptr_t data_addr = (uintptr_t)read_t->buffer->data;
    assert((data_addr % 64) == 0);

    // I dati letti via mmap devono essere perfettamente identici
    float *read_data = read_t->buffer->data;
    for (int i = 0; i < 15; i++) {
        assert(read_data[i] == data[i]);
    }

    tensorDeref(t);
    tensorDeref(read_t); // Questo deve smappare il file tramite munmap
    unlink(filename);

    printf("[TEST IO] test_binary_io superato!\n");
}

// 3. Robustezza ed errori di I/O (devono causare exit(1))
void test_io_errors() {
    printf("[TEST IO] test_io_errors...\n");

    // Lettura file PGM inesistente
    TEST_EXITS({
        tensor *t = readPGM("/tmp/questo_file_sicuramente_non_esiste_pgm.pgm");
        (void)t;
    });
    printf("  readPGM file inesistente: OK\n");

    // Scrittura PGM con tensore 1D (deve fallire per equalDimensions in writePGM)
    TEST_EXITS({
        int32_t shape[] = {10};
        tensor *t = allocTensor(1, shape);
        writePGM(t, "/tmp/should_fail.pgm");
    });
    printf("  writePGM con tensore 1D: OK\n");

    // Lettura file binario inesistente
    TEST_EXITS({
        tensor *t = readBinary("/tmp/questo_file_sicuramente_non_esiste_bin.bin");
        (void)t;
    });
    printf("  readBinary file inesistente: OK\n");

    // Lettura PGM con formato non P5
    const char *bad_pgm = "/tmp/bad_format.pgm";
    FILE *f = fopen(bad_pgm, "w");
    fprintf(f, "P2\n2 2\n255\n0 1 2 3\n"); // P2 non è P5
    fclose(f);
    TEST_EXITS({
        tensor *t = readPGM(bad_pgm);
        (void)t;
    });
    unlink(bad_pgm);
    printf("  readPGM formato non P5: OK\n");

    printf("[TEST IO] test_io_errors superato!\n");
}

int main() {
    printf("=== AVVIO SUITE COMPLETA TEST IO ===\n");
    test_pgm_io();
    test_binary_io();
    test_io_errors();
    printf("=== TUTTI I TEST IO PASSATI CON SUCCESSO! ===\n");
    return 0;
}
