// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#include "token.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Dimensione iniziale dell'array di token (cresce dinamicamente)
#define TOKEN_INITIAL_CAPACITY 64

// Funzioni interne di supporto

// Legge l'intero contenuto di un file in una stringa allocata dinamicamente.
// Ritorna la stringa (da liberare col free), oppure NULL in caso di errore.
static char *readFileContent(const char *fileName) {
	FILE *filePtr = fopen(fileName, "r");
	if (filePtr == NULL) {
		fprintf(stderr, "Errore: impossibile aprire il file sorgente '%s'\n", fileName);
		exit(1);
	}

	// Determina la dimensione del file
	fseek(filePtr, 0, SEEK_END);
	long fileSize = ftell(filePtr);
	rewind(filePtr);

	char *fileContent = malloc((size_t)fileSize + 1);
	if (fileContent == NULL) {
		fprintf(stderr, "Errore: allocazione memoria per il file sorgente\n");
		fclose(filePtr);
		exit(1);
	}

	size_t bytesRead = fread(fileContent, 1, (size_t)fileSize, filePtr);
	fileContent[bytesRead] = '\0';
	fclose(filePtr);
	return fileContent;
}

// Aggiunge un token all'array, crescendo dinamicamente se necessario.
static void appendToken(Token **tokenArrayPtr, size_t *tokenCountPtr, size_t *tokenCapacityPtr, Token tokenToAppend) {
	if (*tokenCountPtr >= *tokenCapacityPtr) {
		*tokenCapacityPtr *= 2;
		*tokenArrayPtr = realloc(*tokenArrayPtr, sizeof(Token) * (*tokenCapacityPtr));
		if (*tokenArrayPtr == NULL) {
			fprintf(stderr, "Errore: riallocazione array token\n");
			exit(1);
		}
	}
	(*tokenArrayPtr)[(*tokenCountPtr)++] = tokenToAppend;
}

// Converte un singolo carattere operatore nel corrispondente TokenType.
// Ritorna TOKEN_UNKNOWN se il carattere non corrisponde a nessun operatore.
static TokenType charToTokenType(char character) {
	switch (character) {
	case '+':
		return TOKEN_ADD;
	case '-':
		return TOKEN_SUB;
	case '*':
		return TOKEN_MUL;
	case '<':
		return TOKEN_LT;
	case '>':
		return TOKEN_GT;
	case '=':
		return TOKEN_EQ;
	case '&':
		return TOKEN_AND;
	case '|':
		return TOKEN_OR;
	case '!':
		return TOKEN_NOT;
	case '$':
		return TOKEN_SELECT;
	case '@':
		return TOKEN_MATMUL;
	case '.':
		return TOKEN_DOT;
	case 'c':
		return TOKEN_CONV2D;
	case 'r':
		return TOKEN_RESHAPE;
	case '_':
		return TOKEN_RAVEL;
	case '#':
		return TOKEN_SHAPE;
	case '?':
		return TOKEN_RANDOM;
	case 'R':
		return TOKEN_RELU;
	case 'm':
		return TOKEN_MIN;
	case 'M':
		return TOKEN_MAX;
	case 'S':
		return TOKEN_REDUCTION_SUM;
	case 'f':
		return TOKEN_FILL;
	case '(':
		return TOKEN_READ_PGM;
	case ')':
		return TOKEN_WRITE_PGM;
	case '{':
		return TOKEN_READ_BINARY;
	case '}':
		return TOKEN_WRITE_BINARY;
	case 'd':
		return TOKEN_DUP;
	case 's':
		return TOKEN_SWAP;
	case 'o':
		return TOKEN_OVER;
	case 'D':
		return TOKEN_DROP;
	case 'p':
		return TOKEN_PRINT;
	default:
		return TOKEN_UNKNOWN;
	}
}

// Parsa la sequenza "[ n1 n2 ... ]" a partire da *pos nel testo.
// *pos deve puntare al carattere DOPO '['.
// Ritorna un Token di tipo TOKEN_TENSOR_LITERAL col tensor* allocato.
// Avanza *pos fino al carattere dopo ']'.
static Token parseTensorLiteral(const char *sourceString, size_t *positionPtr, size_t sourceLength) {
	// Raccoglie i float in un buffer dinamico
	float *valueBuffer = NULL;
	size_t valueCount = 0;
	size_t valueCapacity = 8;
	valueBuffer = malloc(sizeof(float) * valueCapacity);
	if (valueBuffer == NULL) {
		fprintf(stderr, "Errore: allocazione buffer tensore letterale\n");
		exit(1);
	}

	while (*positionPtr < sourceLength) {
		// Salta spazi e newline
		while (*positionPtr < sourceLength && isspace((unsigned char)sourceString[*positionPtr])) {
			(*positionPtr)++;
		}
		if (*positionPtr >= sourceLength)
			break;

		if (sourceString[*positionPtr] == ']') {
			(*positionPtr)++; // consuma ']'
			break;
		}

		// Legge un numero float
		char *endPointer;
		float floatValue = strtof(sourceString + *positionPtr, &endPointer);
		if (endPointer == sourceString + *positionPtr) {
			// Nessun numero letto: carattere inatteso dentro [ ]
			fprintf(stderr, "Errore: carattere inatteso '%c' dentro un tensore letterale\n",
					sourceString[*positionPtr]);
			free(valueBuffer);
			exit(1);
		}
		*positionPtr = (size_t)(endPointer - sourceString);

		// Crescita dinamica del buffer
		if (valueCount >= valueCapacity) {
			valueCapacity *= 2;
			valueBuffer = realloc(valueBuffer, sizeof(float) * valueCapacity);
			if (valueBuffer == NULL) {
				fprintf(stderr, "Errore: riallocazione buffer tensore\n");
				exit(1);
			}
		}
		valueBuffer[valueCount++] = floatValue;
	}

	if (valueCount == 0) {
		fprintf(stderr, "Errore: tensore letterale vuoto [ ]\n");
		free(valueBuffer);
		exit(1);
	}

	// Alloca il tensore 1D con i valori letti
	int32_t tensorShape[1] = {(int32_t)valueCount};
	tensor *resultTensor = allocTensor(1, tensorShape);
	for (size_t i = 0; i < valueCount; i++) {
		resultTensor->buffer->data[i] = valueBuffer[i];
	}
	free(valueBuffer);

	Token resultToken;
	resultToken.type = TOKEN_TENSOR_LITERAL;
	resultToken.value.t = resultTensor;
	return resultToken;
}

// Parsa la stringa "..." a partire da *pos nel testo.
// *pos deve puntare al carattere DOPO il primo '"'.
// Ritorna un Token di tipo TOKEN_STRING con char* allocato.
// Avanza *pos fino al carattere dopo '"' di chiusura.
static Token parseString(const char *sourceString, size_t *positionPtr, size_t sourceLength) {
	size_t startIndex = *positionPtr;
	while (*positionPtr < sourceLength && sourceString[*positionPtr] != '"') {
		(*positionPtr)++;
	}
	if (*positionPtr >= sourceLength) {
		fprintf(stderr, "Errore: stringa non chiusa (manca '\"' di chiusura)\n");
		exit(1);
	}

	size_t stringLength = *positionPtr - startIndex;
	char *stringBuffer = malloc(stringLength + 1);
	if (stringBuffer == NULL) {
		fprintf(stderr, "Errore: allocazione stringa token\n");
		exit(1);
	}
	memcpy(stringBuffer, sourceString + startIndex, stringLength);
	stringBuffer[stringLength] = '\0';

	(*positionPtr)++; // consuma '"' di chiusura

	Token resultToken;
	resultToken.type = TOKEN_STRING;
	resultToken.value.str = stringBuffer;
	return resultToken;
}
    
// Legge l'intero file sorgente TensorForth e produce un array di Token.
// I token sono separati da spazi o ritorni a capo (come da specifica).
// I tensori letterali [ ... ] e le stringhe "..." vengono completamente
// parsati e il loro payload è già disponibile nella union del Token.
// Parametri:
//   fileName      - path del file sorgente
//   outTokenCount - in output: numero di token nell'array ritornato
// Ritorna: array Token* allocato dinamicamente; usa freeTokens() per liberarlo.
Token *tokenize(const char *fileName, size_t *outTokenCount) {
	char *sourceString = readFileContent(fileName);
	size_t sourceLength = strlen(sourceString);
	size_t positionIndex = 0;

	size_t tokenCapacity = TOKEN_INITIAL_CAPACITY;
	size_t tokenCount = 0;
	Token *tokenArray = malloc(sizeof(Token) * tokenCapacity);
	if (tokenArray == NULL) {
		fprintf(stderr, "Errore: allocazione array token\n");
		free(sourceString);
		exit(1);
	}

	while (positionIndex < sourceLength) {
		// 1. Salta spazi e ritorni a capo
		if (isspace((unsigned char)sourceString[positionIndex])) {
			positionIndex++;
			continue;
		}

		char character = sourceString[positionIndex];

		// 2. Tensore letterale: inizia con '[ ' (spazio obbligatorio dopo '[')
		if (character == '[') {
			// Controlla che il carattere successivo sia uno spazio (specifica)
			if (positionIndex + 1 >= sourceLength || !isspace((unsigned char)sourceString[positionIndex + 1])) {
				fprintf(stderr, "Errore di sintassi: '[' deve essere seguito da uno spazio\n");
				free(sourceString);
				freeTokens(tokenArray, tokenCount);
				exit(1);
			}
			positionIndex++; // consuma '['
			Token currentToken = parseTensorLiteral(sourceString, &positionIndex, sourceLength);
			appendToken(&tokenArray, &tokenCount, &tokenCapacity, currentToken);
			continue;
		}

		// 3. Stringa tra doppi apici
		if (character == '"') {
			positionIndex++; // consuma '"' di apertura
			Token currentToken = parseString(sourceString, &positionIndex, sourceLength);
			appendToken(&tokenArray, &tokenCount, &tokenCapacity, currentToken);
			continue;
		}

		// Supporta sia '|' che '\|' (typo LaTeX comune nelle specifiche del PDF) per l'operatore OR
		if (character == '\\' && positionIndex + 1 < sourceLength && sourceString[positionIndex + 1] == '|') {
			Token currentToken;
			currentToken.type = TOKEN_OR;
			currentToken.value.t = NULL;
			appendToken(&tokenArray, &tokenCount, &tokenCapacity, currentToken);
			positionIndex += 2;
			continue;
		}

		// 4. Operatore a singolo carattere
		TokenType tokenType = charToTokenType(character);
		if (tokenType != TOKEN_UNKNOWN) {
			Token currentToken;
			currentToken.type = tokenType;
			currentToken.value.t = NULL; // nessun payload per gli operatori
			appendToken(&tokenArray, &tokenCount, &tokenCapacity, currentToken);
			positionIndex++;
			continue;
		}

		// 5. Token non riconosciuto
		fprintf(stderr, "Errore: token non riconosciuto '%c' (posizione %zu)\n", character, positionIndex);
		free(sourceString);
		freeTokens(tokenArray, tokenCount);
		exit(1);
	}

	free(sourceString);
	*outTokenCount = tokenCount;
	return tokenArray;
}

// Libera un array di Token prodotto da tokenize().
// Per TOKEN_TENSOR_LITERAL chiama tensorDeref sul tensore allocato.
// Per TOKEN_STRING chiama free sulla stringa allocata.
void freeTokens(Token *tokenArray, size_t tokenCount) {
	if (tokenArray == NULL)
		return;
	for (size_t i = 0; i < tokenCount; i++) {
		if (tokenArray[i].type == TOKEN_TENSOR_LITERAL) {
			tensorDeref(tokenArray[i].value.t);
		} else if (tokenArray[i].type == TOKEN_STRING) {
			free(tokenArray[i].value.str);
		}
	}
	free(tokenArray);
}
