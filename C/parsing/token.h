// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#ifndef TOKEN_H
#define TOKEN_H

#include "../tensor/tensor.h"
#include <stddef.h>

// Enum di tutti i tipi di token del linguaggio TensorForth.
// I token operatori sono identificati dal loro simbolo;
// i token speciali (tensore letterale e stringa) portano un payload.
typedef enum {
	// Letterali (portano un payload nella union)
	TOKEN_TENSOR_LITERAL, // [ n1 n2 ... ] -> payload: tensor*
	TOKEN_STRING,		  // "nome.pgm" -> payload: char*

	// Operatori aritmetici
	TOKEN_ADD, // +
	TOKEN_SUB, // -
	TOKEN_MUL, // *

	// Operatori di comparazione
	TOKEN_LT, // <
	TOKEN_GT, // >
	TOKEN_EQ, // =

	// Operatori logici
	TOKEN_AND, // &
	TOKEN_OR,  // |
	TOKEN_NOT, // !

	// Selezione
	TOKEN_SELECT, // $

	// Operatori su tensori
	TOKEN_MATMUL, // @
	TOKEN_DOT,	  // .
	TOKEN_CONV2D, // c

	// Forma e dimensione
	TOKEN_RESHAPE, // r
	TOKEN_RAVEL,   // _
	TOKEN_SHAPE,   // #

	// Generazione, riduzione, riempimento
	TOKEN_RANDOM,		 // ?
	TOKEN_RELU,			 // R
	TOKEN_MIN,			 // m
	TOKEN_MAX,			 // M
	TOKEN_REDUCTION_SUM, // S
	TOKEN_FILL,			 // f

	// --- I/O ---
	TOKEN_READ_PGM,		// (
	TOKEN_WRITE_PGM,	// )
	TOKEN_READ_BINARY,	// {
	TOKEN_WRITE_BINARY, // }

	// Manipolazione stack
	TOKEN_DUP,	// d
	TOKEN_SWAP, // s
	TOKEN_OVER, // o
	TOKEN_DROP, // D

	// Utilità
	TOKEN_PRINT, // p

	// Errore
	TOKEN_UNKNOWN
} TokenType;

// Struttura che rappresenta un singolo token.
// Per gli operatori il payload è ignorato.
// Per TOKEN_TENSOR_LITERAL il payload è un tensor* già allocato.
// Per TOKEN_STRING il payload è una stringa allocata dinamicamente.
typedef struct {
	TokenType type;
	union {
		tensor *t; // usato da TOKEN_TENSOR_LITERAL
		char *str; // usato da TOKEN_STRING
	} value;
} Token;

// Legge l'intero file sorgente e produce un array di Token.
// Parametri:
//   fileName      - path del file sorgente TensorForth
//   outTokenCount - in output: numero di token prodotti
// Ritorna un array di Token allocato dinamicamente (da liberare con freeTokens).
// In caso di errore stampa su stderr e chiama exit(1).
Token *tokenize(const char *fileName, size_t *outTokenCount);

// Libera un array di Token prodotto da tokenize().
// Per TOKEN_TENSOR_LITERAL chiama tensorDeref; per TOKEN_STRING chiama free.
void freeTokens(Token *tokenArray, size_t tokenCount);

#endif
