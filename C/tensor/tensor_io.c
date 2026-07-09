// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

// tensor_io.c - Operazioni di I/O sui tensori:
//   ReadPGM, WritePGM (immagini in formato PGM binario P5),
//   ReadBinary, WriteBinary (formato binario allineato a 64 byte con mmap).

#include "tensor.h"
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// Lettura e scrittura PGM (formato P5 - scala di grigi binario)

// Legge un'immagine PGM binaria (P5) e la restituisce come tensore 2D.
// I valori dei pixel vengono normalizzati in [0.0, 1.0].
// Input: path del file. Output: tensore 2D (height x width).
tensor *readPGM(const char *fileName) {
	FILE *filePtr = fopen(fileName, "rb");
	if (filePtr == NULL) {
		fprintf(stderr, "Errore: impossibile aprire il file PGM '%s'\n", fileName);
		exit(1);
	}

	// Verifica magic number P5
	char fileHeader[16];
	if (fscanf(filePtr, "%15s", fileHeader) != 1 || strcmp(fileHeader, "P5") != 0) {
		fprintf(stderr, "Errore: formato PGM non valido (atteso P5, trovato '%s')\n", fileHeader);
		fclose(filePtr);
		exit(1);
	}

	// Salta commenti e spazi bianchi
	int character = fgetc(filePtr);
	while (character != EOF) {
		if (character == '#') {
			while ((character = fgetc(filePtr)) != '\n' && character != EOF)
				;
		} else if (character != ' ' && character != '\t' && character != '\r' && character != '\n') {
			ungetc(character, filePtr);
			break;
		}
		character = fgetc(filePtr); // ripristinare il primo elemento utile
	}

	// Legge dimensioni e valore massimo
	int32_t imageWidth = 0, imageHeight = 0, maxPixelValue = 0;
	if (fscanf(filePtr, "%d %d %d", &imageWidth, &imageHeight, &maxPixelValue) != 3) {
		fprintf(stderr, "Errore: intestazione PGM non valida in '%s'\n", fileName);
		fclose(filePtr);
		exit(1);
	}
	fgetc(filePtr); // consuma il singolo whitespace dopo maxPixelValue

	int32_t tensorShape[] = {imageHeight, imageWidth};
	tensor *resultTensor = allocTensor(2, tensorShape);

	// Legge i pixel in un buffer temporaneo e li normalizza
	uint8_t *pixelBuffer = malloc((size_t)(imageWidth * imageHeight));
	if (pixelBuffer == NULL) {
		fprintf(stderr, "Errore: allocazione buffer pixel PGM\n");
		fclose(filePtr);
		exit(1);
	}

	if (fread(pixelBuffer, 1, (size_t)(imageWidth * imageHeight), filePtr) !=
		(size_t)(imageWidth * imageHeight)) {
		fprintf(stderr, "Errore: lettura pixel incompleta in '%s'\n", fileName);
		free(pixelBuffer);
		fclose(filePtr);
		exit(1);
	}
	fclose(filePtr);

	float *tensorData = resultTensor->buffer->data;
	for (int32_t pixelIndex = 0; pixelIndex < imageWidth * imageHeight; pixelIndex++)
		tensorData[pixelIndex] = (float)pixelBuffer[pixelIndex] / (float)maxPixelValue; // normalizzazione dei pixel

	free(pixelBuffer);
	return resultTensor;
}

// Scrive un tensore 2D come immagine PGM binaria (P5).
// Valori < 0 sono trattati come 0, valori > 1 come 1; rimappati in [0, 255].
// Input: tensore 2D e path di output. Ritorna 1 su successo, 0 su errore.
int writePGM(tensor *tensorToWrite, const char *fileName) {
	equalDimensions(tensorToWrite->dimensionOfTensor, 2);

	FILE *filePtr = fopen(fileName, "wb");
	if (filePtr == NULL) {
		fprintf(stderr, "Errore: impossibile creare '%s'\n", fileName);
		exit(1);
	}

	int32_t imageHeight = tensorToWrite->shape[0];
	int32_t imageWidth = tensorToWrite->shape[1];
	fprintf(filePtr, "P5\n%d %d\n255\n", imageWidth, imageHeight);

	float *tensorData = tensorToWrite->buffer->data;
	uint8_t *pixelBuffer = malloc((size_t)(imageWidth * imageHeight));
	if (pixelBuffer == NULL) {
		fprintf(stderr, "Errore: allocazione buffer pixel per scrittura PGM\n");
		fclose(filePtr);
		exit(1);
	}

	for (int32_t pixelIndex = 0; pixelIndex < imageWidth * imageHeight; pixelIndex++) {
		float pixelValue = tensorData[pixelIndex];
		if (pixelValue < 0.0f)
			pixelValue = 0.0f;
		if (pixelValue > 1.0f)
			pixelValue = 1.0f;
		pixelBuffer[pixelIndex] = (uint8_t)(pixelValue * 255.0f + 0.5f);
	}

	size_t bytesWritten = fwrite(pixelBuffer, 1, (size_t)(imageWidth * imageHeight), filePtr);
	free(pixelBuffer);
	fclose(filePtr);

	if (bytesWritten != (size_t)(imageWidth * imageHeight)) {
		fprintf(stderr, "Errore: scrittura pixel incompleta in '%s'\n",
				fileName);
		exit(1);
	}

	return 1;
}

// Lettura e scrittura binaria con mmap

// Legge un tensore da file binario usando mmap (zero-copy: nessuna copia dei
// dati). Il file deve seguire il formato on_disk_tensor con data_offset = 64.
// Input: path del file. Output: tensore con buffer backed da mmap.
tensor *readBinary(const char *fileName) {
	int fileDescriptor = open(fileName, O_RDONLY);
	if (fileDescriptor < 0) {
		fprintf(stderr, "Errore: impossibile aprire '%s'\n", fileName);
		exit(1);
	}

	struct stat fileMetadata; // -> metadati di un file
	if (fstat(fileDescriptor, &fileMetadata) < 0) {
		fprintf(stderr, "Errore: fstat su '%s'\n", fileName);
		close(fileDescriptor);
		exit(1);
	}
	size_t fileSize = (size_t)fileMetadata.st_size;

	// Legge l'header dal file
	struct on_disk_tensor diskHeader;
	if (read(fileDescriptor, &diskHeader, sizeof(diskHeader)) != (ssize_t)sizeof(diskHeader)) {
		fprintf(stderr, "Errore: lettura header da '%s'\n", fileName);
		close(fileDescriptor);
		exit(1);
	}

	// Mappa l'intero file in memoria
	void *mappedAddress = mmap(NULL, fileSize, PROT_READ, MAP_SHARED, fileDescriptor, 0);
	if (mappedAddress == MAP_FAILED) {
		fprintf(stderr, "Errore: mmap fallito per '%s'\n", fileName);
		close(fileDescriptor);
		exit(1);
	}
	close(fileDescriptor);

	// Costruisce la struttura logica del tensore senza copiare i dati
	tensor *resultTensor = malloc(sizeof(tensor));
	if (resultTensor == NULL) {
		fprintf(stderr, "Errore: allocazione tensor\n");
		munmap(mappedAddress, fileSize);
		exit(1);
	}

	resultTensor->dimensionOfTensor = diskHeader.ndim;
	resultTensor->referenceCount = 1;
	resultTensor->shape[0] = diskHeader.shape[0];
	resultTensor->shape[1] = diskHeader.shape[1];

	resultTensor->buffer = malloc(sizeof(tensor_buffer));
	if (resultTensor->buffer == NULL) {
		fprintf(stderr, "Errore: allocazione tensor_buffer\n");
		free(resultTensor);
		munmap(mappedAddress, fileSize);
		exit(1);
	}

	resultTensor->buffer->referenceCount = 1;
	resultTensor->buffer->isMMapped = 1;
	resultTensor->buffer->mmapAddress = mappedAddress;
	resultTensor->buffer->mmapLength = fileSize;
	resultTensor->buffer->data = (float *)((char *)mappedAddress + diskHeader.data_offset);

	return resultTensor;
}

// Scrive un tensore su file nel formato binario allineato a 64 byte.
// Struttura: [header on_disk_tensor][padding fino a 64 byte][dati float].
// Input: tensore e path di output. Ritorna 1 su successo, 0 su errore.
int writeBinary(tensor *inputTensor, const char *fileName) {
	FILE *binaryFile = fopen(fileName, "wb");
	if (binaryFile == NULL) {
		fprintf(stderr, "Errore: impossibile creare '%s'\n", fileName);
		exit(1);
	}

	struct on_disk_tensor diskHeader;
	diskHeader.shape[0] = inputTensor->shape[0];
	diskHeader.shape[1] = inputTensor->shape[1];
	diskHeader.ndim = inputTensor->dimensionOfTensor;
	diskHeader.data_offset = 64; // offset minimo per allineamento a 64 byte (MAX_DIM=2)

	if (fwrite(&diskHeader, sizeof(diskHeader), 1, binaryFile) != 1) {
		fprintf(stderr, "Errore: scrittura header binario\n");
		fclose(binaryFile);
		exit(1);
	}

	// Padding fino a 64 byte
	size_t headerSize = sizeof(diskHeader);
	size_t paddingSize = 64 - headerSize;
	uint8_t paddingBuffer[64] = {0};
	if (fwrite(paddingBuffer, 1, paddingSize, binaryFile) != paddingSize) {
		fprintf(stderr, "Errore: scrittura padding\n");
		fclose(binaryFile);
		exit(1);
	}

	// Dati del tensore
	int32_t totalElements = getTotalElements(inputTensor);
	size_t elementsWritten = fwrite(inputTensor->buffer->data, sizeof(float), (size_t)totalElements, binaryFile);
	fclose(binaryFile);

	if (elementsWritten != (size_t)totalElements) {
		fprintf(stderr, "Errore: scrittura dati incompleta in '%s'\n", fileName);
		exit(1);
	}

	return 1;
}
