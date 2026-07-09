# Nome: Filippo
# Cognome: Morello
# Matricola: SM3201475

import json
import os
from renderingEngine.exceptions.customExceptions import invalidPaletteException

class Palette:
    # Rappresenta la palette di 16 colori indicizzati.
    
    def __init__(self, jsonFilePath=""):
        self.colorsList = []
        if jsonFilePath:
            self.loadPalette(jsonFilePath)
            
    def loadPalette(self, jsonFilePath):
        # Verifica se il file esiste
        if not os.path.exists(jsonFilePath):
            raise invalidPaletteException(f"Il file della palette non esiste: {jsonFilePath}")
            
        try:
            with open(jsonFilePath, "r") as fileStream:
                parsedData = json.load(fileStream)
        except json.JSONDecodeError as decodeError:
            raise invalidPaletteException(f"Errore nel parsing del JSON della palette: {decodeError}")
        except Exception as errorObject:
            raise invalidPaletteException(f"Errore nella lettura della palette: {errorObject}")
            
        self.validateAndSetPalette(parsedData)
        
    def validateAndSetPalette(self, paletteData):
        # Valida che sia una lista
        if not isinstance(paletteData, list):
            raise invalidPaletteException("Il file JSON della palette deve contenere una lista.")
            
        # Devono esserci esattamente 16 colori
        if len(paletteData) != 16:
            raise invalidPaletteException(f"La palette deve contenere esattamente 16 colori, trovati {len(paletteData)}.")
            
        validatedColors = []
        for colorIndex, rawColor in enumerate(paletteData):
            if not isinstance(rawColor, list):
                raise invalidPaletteException(f"Il colore all'indice {colorIndex} non e' una lista.")
                
            if len(rawColor) != 3:
                raise invalidPaletteException(f"Il colore all'indice {colorIndex} deve avere 3 componenti RGB, trovate {len(rawColor)}.")
                
            validatedRGB = []
            for componentIndex, channelValue in enumerate(rawColor):
                if not isinstance(channelValue, int) or isinstance(channelValue, bool):
                    raise invalidPaletteException(
                        f"Il colore all'indice {colorIndex}, componente {componentIndex} non e' un intero: {channelValue}"
                    )
                if channelValue < 0 or channelValue > 255:
                    raise invalidPaletteException(
                        f"Il colore all'indice {colorIndex}, component {componentIndex} e' fuori range (0-255): {channelValue}"
                    )
                validatedRGB.append(channelValue)
                
            validatedColors.append((validatedRGB[0], validatedRGB[1], validatedRGB[2]))
            
        self.colorsList = validatedColors
        
    def resolveColor(self, colorIndex):
        # Risolve l'indice del colore in una terna RGB
        if colorIndex < 0 or colorIndex >= len(self.colorsList):
            raise invalidPaletteException(f"Indice colore {colorIndex} fuori range (0-15).")
        return self.colorsList[colorIndex]
