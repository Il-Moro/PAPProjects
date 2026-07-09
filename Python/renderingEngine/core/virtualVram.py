# Nome: Filippo
# Cognome: Morello
# Matricola: SM3201475

import os
import numpy as np
from renderingEngine.exceptions.customExceptions import invalidVramException

# Caricamento e decodifica delle texture (tile e sprite) dai file binari
class VirtualVRAM:
    
    def __init__(self):
        # Matrici di indici dei colori (256x256 pixel)
        self.tilesMatrix = np.zeros((256, 256), dtype=np.uint8)
        self.spritesMatrix = np.zeros((256, 256), dtype=np.uint8)
        
    def loadTileSheet(self, tilesBinPath):
        # Carica il tile sheet
        self.tilesMatrix = self.decodeVramBinaryFile(tilesBinPath)
        
    def loadSpriteSheet(self, spritesBinPath):
        # Carica lo sprite sheet
        self.spritesMatrix = self.decodeVramBinaryFile(spritesBinPath)
        
    def decodeVramBinaryFile(self, binaryFilePath):
        # Verifica se il file esiste
        if not os.path.exists(binaryFilePath):
            raise invalidVramException(f"Il file binario non esiste: {binaryFilePath}")
            
        # Dimensione del file deve essere esattamente di 32768 byte (256x256 / 2)
        if os.path.getsize(binaryFilePath) != 32768:
            raise invalidVramException(
                f"Dimensione del file binario non corretta: {binaryFilePath}. "
                f"Attesi esattamente 32768 byte, trovati {os.path.getsize(binaryFilePath)}."
            )
            
        try:
            packedBytes = np.fromfile(binaryFilePath, dtype=np.uint8)
            
            # Estrae i nibble (ogni byte contiene 2 pixel da 4 bit)
            highNibbles = (packedBytes >> 4) & 0x0F
            lowNibbles = packedBytes & 0x0F
            
            # Interseca i pixel alti e bassi per preservare l'ordine
            decodedPixels = np.empty(len(packedBytes) * 2, dtype=np.uint8)
            decodedPixels[0::2] = highNibbles
            decodedPixels[1::2] = lowNibbles
            
            # Riassegna la forma a 256x256
            return decodedPixels.reshape((256, 256))
            
        except Exception as errorObject:
            raise invalidVramException(f"Errore nella decodifica del file binario: {errorObject}")
