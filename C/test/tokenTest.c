// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#include "token.h"
#include "tensor.h"

// Crea un file temporaneo con il contenuto dato e ne ritorna il path.
static const char *writeTmp(const char *fileName, const char *content) {
    FILE *filePtr = fopen(fileName, "w");
    assert(filePtr != NULL);
    fputs(content, filePtr);
    fclose(filePtr);
    return fileName;
}

// --------------------------------------------------------------------------
// 1. Tokenizzazione di operatori singoli
// --------------------------------------------------------------------------
void testSingleOperators() {
    printf("[TEST] testSingleOperators...\n");

    const char *sourceString =
        "+ - * < > = & | ! $ @ . c r _ # ? R m M S f ( ) { } d s o D p \\|";
    writeTmp("/tmp/tf_ops.tf", sourceString);

    size_t tokenCount = 0;
    Token *tokenArray = tokenize("/tmp/tf_ops.tf", &tokenCount);
    assert(tokenCount == 32);

    TokenType all[] = {
        TOKEN_ADD, TOKEN_SUB, TOKEN_MUL,
        TOKEN_LT, TOKEN_GT, TOKEN_EQ,
        TOKEN_AND, TOKEN_OR, TOKEN_NOT,
        TOKEN_SELECT,
        TOKEN_MATMUL, TOKEN_DOT, TOKEN_CONV2D,
        TOKEN_RESHAPE, TOKEN_RAVEL, TOKEN_SHAPE,
        TOKEN_RANDOM, TOKEN_RELU, TOKEN_MIN, TOKEN_MAX,
        TOKEN_REDUCTION_SUM, TOKEN_FILL,
        TOKEN_READ_PGM, TOKEN_WRITE_PGM,
        TOKEN_READ_BINARY, TOKEN_WRITE_BINARY,
        TOKEN_DUP, TOKEN_SWAP, TOKEN_OVER, TOKEN_DROP,
        TOKEN_PRINT, TOKEN_OR
    };
    for (size_t i = 0; i < tokenCount; i++) {
        assert(tokenArray[i].type == all[i]);
    }

    freeTokens(tokenArray, tokenCount);
    printf("[TEST] testSingleOperators superato!\n");
}

// --------------------------------------------------------------------------
// 2. Tokenizzazione di tensori letterali 1D e 2D
// --------------------------------------------------------------------------
void testTensorLiterals() {
    printf("[TEST] testTensorLiterals...\n");

    // Tensore 1D con 3 elementi
    writeTmp("/tmp/tf_lit.tf", "[ 1.0 2.5 -3.0 ]");
    size_t tokenCount = 0;
    Token *tokenArray = tokenize("/tmp/tf_lit.tf", &tokenCount);
    assert(tokenCount == 1);
    assert(tokenArray[0].type == TOKEN_TENSOR_LITERAL);
    tensor *resultTensor = tokenArray[0].value.t;
    assert(resultTensor->dimensionOfTensor == 1);
    assert(resultTensor->shape[0] == 3);
    assert(resultTensor->buffer->data[0] == 1.0f);
    assert(resultTensor->buffer->data[1] == 2.5f);
    assert(resultTensor->buffer->data[2] == -3.0f);
    freeTokens(tokenArray, tokenCount);

    // Tensore con un solo elemento
    writeTmp("/tmp/tf_lit2.tf", "[ 42 ]");
    tokenArray = tokenize("/tmp/tf_lit2.tf", &tokenCount);
    assert(tokenCount == 1);
    assert(tokenArray[0].value.t->buffer->data[0] == 42.0f);
    freeTokens(tokenArray, tokenCount);

    // Più tensori letterali nella stessa sorgente
    writeTmp("/tmp/tf_lit3.tf", "[ 10 20 ] [ 30 40 50 ]");
    tokenArray = tokenize("/tmp/tf_lit3.tf", &tokenCount);
    assert(tokenCount == 2);
    assert(tokenArray[0].value.t->shape[0] == 2);
    assert(tokenArray[1].value.t->shape[0] == 3);
    freeTokens(tokenArray, tokenCount);

    printf("[TEST] testTensorLiterals superato!\n");
}

// --------------------------------------------------------------------------
// 3. Tokenizzazione di stringhe
// --------------------------------------------------------------------------
void testStringLiterals() {
    printf("[TEST] testStringLiterals...\n");

    writeTmp("/tmp/tf_str.tf", "\"immagine.pgm\"");
    size_t tokenCount = 0;
    Token *tokenArray = tokenize("/tmp/tf_str.tf", &tokenCount);
    assert(tokenCount == 1);
    assert(tokenArray[0].type == TOKEN_STRING);
    assert(strcmp(tokenArray[0].value.str, "immagine.pgm") == 0);
    freeTokens(tokenArray, tokenCount);

    // Stringa + operatore
    writeTmp("/tmp/tf_str2.tf", "\"file.bin\" {");
    tokenArray = tokenize("/tmp/tf_str2.tf", &tokenCount);
    assert(tokenCount == 2);
    assert(tokenArray[0].type == TOKEN_STRING);
    assert(tokenArray[1].type == TOKEN_READ_BINARY);
    freeTokens(tokenArray, tokenCount);

    printf("[TEST] testStringLiterals superato!\n");
}

// --------------------------------------------------------------------------
// 4. Tokenizzazione di un programma completo dal PDF
// --------------------------------------------------------------------------
void testFullProgram() {
    printf("[TEST] testFullProgram...\n");

    // Programma: [ 5 5 ] d + p
    writeTmp("/tmp/tf_prog.tf", "[ 5 5 ] d + p");
    size_t tokenCount = 0;
    Token *tokenArray = tokenize("/tmp/tf_prog.tf", &tokenCount);
    assert(tokenCount == 4);
    assert(tokenArray[0].type == TOKEN_TENSOR_LITERAL);
    assert(tokenArray[0].value.t->shape[0] == 2);
    assert(tokenArray[0].value.t->buffer->data[0] == 5.0f);
    assert(tokenArray[1].type == TOKEN_DUP);
    assert(tokenArray[2].type == TOKEN_ADD);
    assert(tokenArray[3].type == TOKEN_PRINT);
    freeTokens(tokenArray, tokenCount);

    printf("[TEST] testFullProgram superato!\n");
}

// --------------------------------------------------------------------------
// 5. File vuoto → 0 token
// --------------------------------------------------------------------------
void testEmptyFile() {
    printf("[TEST] testEmptyFile...\n");
    writeTmp("/tmp/tf_empty.tf", "");
    size_t tokenCount = 0;
    Token *tokenArray = tokenize("/tmp/tf_empty.tf", &tokenCount);
    assert(tokenCount == 0);
    freeTokens(tokenArray, tokenCount);
    printf("[TEST] testEmptyFile superato!\n");
}

// --------------------------------------------------------------------------
// 6. Spazi e ritorni a capo multipli vengono ignorati
// --------------------------------------------------------------------------
void testWhitespaceHandling() {
    printf("[TEST] testWhitespaceHandling...\n");
    writeTmp("/tmp/tf_ws.tf", "\n\n  +  \n  -  \n\n");
    size_t tokenCount = 0;
    Token *tokenArray = tokenize("/tmp/tf_ws.tf", &tokenCount);
    assert(tokenCount == 2);
    assert(tokenArray[0].type == TOKEN_ADD);
    assert(tokenArray[1].type == TOKEN_SUB);
    freeTokens(tokenArray, tokenCount);
    printf("[TEST] testWhitespaceHandling superato!\n");
}

// --------------------------------------------------------------------------
// 7. File non esistente → exit(1)
// --------------------------------------------------------------------------
void testMissingFileExits() {
    printf("[TEST] testMissingFileExits...\n");
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        freopen("/dev/null", "w", stderr);
        size_t tokenCount = 0;
        tokenize("/tmp/questo_file_non_esiste_xyz.tf", &tokenCount);
        exit(0); // non dovrebbe arrivare qui
    } else {
        int st; waitpid(pid, &st, 0);
        assert(WIFEXITED(st) && WEXITSTATUS(st) == 1);
    }
    printf("[TEST] testMissingFileExits superato!\n");
}

// --------------------------------------------------------------------------
// 8. freeTokens libera correttamente (nessun leak rilevabile a runtime)
// --------------------------------------------------------------------------
void testFreeTokens() {
    printf("[TEST] testFreeTokens...\n");
    writeTmp("/tmp/tf_free.tf", "[ 1 2 3 ] \"file.pgm\" +");
    size_t tokenCount = 0;
    Token *tokenArray = tokenize("/tmp/tf_free.tf", &tokenCount);
    assert(tokenCount == 3);
    // Il tensore allocato dal lexer ha refcount = 1
    tensor *resultTensor = tokenArray[0].value.t;
    assert(resultTensor->referenceCount == 1);
    freeTokens(tokenArray, tokenCount); // deve chiamare tensorDeref su t → libera
    // Non possiamo controllare il refcount dopo (memoria liberata), ma se non crasha va bene
    printf("[TEST] testFreeTokens superato!\n");
}

// --------------------------------------------------------------------------
// 9. Gestione errori del Lexer (devono causare exit(1))
// --------------------------------------------------------------------------
void testLexerErrors() {
    printf("[TEST] testLexerErrors...\n");

    // 1. Parentesi quadra non seguita da spazio: [1 2 ]
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        freopen("/dev/null", "w", stderr);
        writeTmp("/tmp/tf_err1.tf", "[1 2 3 ]");
        size_t tokenCount = 0;
        tokenize("/tmp/tf_err1.tf", &tokenCount);
        exit(0);
    } else {
        int st; waitpid(pid, &st, 0);
        assert(WIFEXITED(st) && WEXITSTATUS(st) == 1);
        unlink("/tmp/tf_err1.tf");
    }
    printf("  '[' non seguito da spazio: OK\n");

    // 2. Stringa non chiusa
    pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        freopen("/dev/null", "w", stderr);
        writeTmp("/tmp/tf_err2.tf", "\"stringa non chiusa");
        size_t tokenCount = 0;
        tokenize("/tmp/tf_err2.tf", &tokenCount);
        exit(0);
    } else {
        int st; waitpid(pid, &st, 0);
        assert(WIFEXITED(st) && WEXITSTATUS(st) == 1);
        unlink("/tmp/tf_err2.tf");
    }
    printf("  Stringa non chiusa: OK\n");

    // 3. Carattere inatteso dentro [ ]
    pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        freopen("/dev/null", "w", stderr);
        writeTmp("/tmp/tf_err3.tf", "[ 1.0 abc 2.0 ]");
        size_t tokenCount = 0;
        tokenize("/tmp/tf_err3.tf", &tokenCount);
        exit(0);
    } else {
        int st; waitpid(pid, &st, 0);
        assert(WIFEXITED(st) && WEXITSTATUS(st) == 1);
        unlink("/tmp/tf_err3.tf");
    }
    printf("  Carattere non numerico dentro tensore: OK\n");

    // 4. Token sconosciuto (es. %)
    pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        freopen("/dev/null", "w", stderr);
        writeTmp("/tmp/tf_err4.tf", "[ 1 2 3 ] % p");
        size_t tokenCount = 0;
        tokenize("/tmp/tf_err4.tf", &tokenCount);
        exit(0);
    } else {
        int st; waitpid(pid, &st, 0);
        assert(WIFEXITED(st) && WEXITSTATUS(st) == 1);
        unlink("/tmp/tf_err4.tf");
    }
    printf("  Token non riconosciuto: OK\n");

    printf("[TEST] testLexerErrors superato!\n");
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------
int main() {
    printf("=== AVVIO SUITE COMPLETA TEST TOKEN/PARSER ===\n");
    testTensorLiterals();
    testStringLiterals();
    testSingleOperators();
    testFullProgram();
    testEmptyFile();
    testWhitespaceHandling();
    testMissingFileExits();
    testFreeTokens();
    testLexerErrors();
    printf("=== TUTTI I TEST TOKEN PASSATI CON SUCCESSO! ===\n");
    return 0;
}
