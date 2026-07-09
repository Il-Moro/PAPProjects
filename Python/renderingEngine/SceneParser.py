# Nome: Filippo
# Cognome: Morello
# Matricola: SM3201475

import json
import os
import numpy as np
from renderingEngine.customExceptions import invalidSceneException

# Lettura e validazione dei parametri della scena da file JSON
class SceneParser:
    
    def __init__(self, sceneJsonPath=""):
        self.transparentIndex = 0
        self.tileMap = np.zeros((15, 20), dtype=np.uint8)
        self.spritesList = []
        
        if sceneJsonPath:
            self.loadScene(sceneJsonPath)
            
    def loadScene(self, sceneJsonPath):
        # Controlla se il file esiste
        if not os.path.exists(sceneJsonPath):
            raise invalidSceneException(f"Il file della scena non esiste: {sceneJsonPath}")
            
        try:
            with open(sceneJsonPath, "r") as fileStream:
                parsedData = json.load(fileStream)
        except json.JSONDecodeError as decodeError:
            raise invalidSceneException(f"Errore nel parsing del JSON della scena: {decodeError}")
        except Exception as errorObject:
            raise invalidSceneException(f"Errore nella lettura del file della scena: {errorObject}")
            
        self.validateAndSetScene(parsedData)
        
    def validateAndSetScene(self, sceneData):
        if not isinstance(sceneData, dict):
            raise invalidSceneException("Il JSON della scena deve contenere un dizionario principale.")
            
        # 1. Valida l'indice di trasparenza (0-15)
        if "transparent_index" not in sceneData:
            raise invalidSceneException("Chiave 'transparent_index' mancante.")
        transparentIndexVal = sceneData["transparent_index"]
        if not isinstance(transparentIndexVal, int) or isinstance(transparentIndexVal, bool):
            raise invalidSceneException(f"transparent_index deve essere un intero.")
        if transparentIndexVal < 0 or transparentIndexVal > 15:
            raise invalidSceneException(f"transparent_index fuori range (0-15): {transparentIndexVal}")
        self.transparentIndex = transparentIndexVal
        
        # 2. Valida la mappa dei tile del fondale (15x20)
        if "tile_map" not in sceneData:
            raise invalidSceneException("Chiave 'tile_map' mancante.")
        tileMapVal = sceneData["tile_map"]
        if not isinstance(tileMapVal, list):
            raise invalidSceneException("tile_map deve essere una lista bidimensionale.")
        if len(tileMapVal) != 15:
            raise invalidSceneException(f"tile_map deve contenere 15 righe, ricevute: {len(tileMapVal)}")
            
        tempTileMap = np.zeros((15, 20), dtype=np.uint8)
        for rowIndex, rowList in enumerate(tileMapVal):
            if not isinstance(rowList, list):
                raise invalidSceneException(f"La riga {rowIndex} di tile_map non e' una lista.")
            if len(rowList) != 20:
                raise invalidSceneException(f"La riga {rowIndex} di tile_map deve avere 20 elementi, trovati {len(rowList)}")
                
            for colIndex, tileId in enumerate(rowList):
                if not isinstance(tileId, int) or isinstance(tileId, bool):
                    raise invalidSceneException(f"Il valore in riga {rowIndex}, col {colIndex} non e' un intero.")
                if tileId < 0 or tileId > 63:
                    raise invalidSceneException(f"Il valore in riga {rowIndex}, col {colIndex} e' fuori range (0-63): {tileId}")
                tempTileMap[rowIndex, colIndex] = tileId
        self.tileMap = tempTileMap
        
        # 3. Valida la lista degli sprite
        if "sprites" not in sceneData:
            raise invalidSceneException("Chiave 'sprites' mancante.")
        spritesVal = sceneData["sprites"]
        if not isinstance(spritesVal, list):
            raise invalidSceneException("sprites deve essere una lista.")
            
        validatedSpritesList = []
        for spriteIndex, spriteDict in enumerate(spritesVal):
            if not isinstance(spriteDict, dict):
                raise invalidSceneException(f"Lo sprite all'indice {spriteIndex} non e' un dizionario.")
                
            requiredSpriteKeys = {"id", "x", "y", "flip_h", "flip_v", "rotation"}
            missingKeys = requiredSpriteKeys - spriteDict.keys()
            if missingKeys:
                raise invalidSceneException(f"Lo sprite all'indice {spriteIndex} ha chiavi mancanti: {missingKeys}")
                
            spriteId = spriteDict["id"]
            if not isinstance(spriteId, int) or isinstance(spriteId, bool):
                raise invalidSceneException(f"id dello sprite {spriteIndex} deve essere un intero.")
            if spriteId < 0 or spriteId > 15:
                raise invalidSceneException(f"id dello sprite {spriteIndex} fuori range (0-15): {spriteId}")
                
            spriteX = spriteDict["x"]
            if not isinstance(spriteX, int) or isinstance(spriteX, bool):
                raise invalidSceneException(f"Coordinata x dello sprite {spriteIndex} deve essere un intero.")
                
            spriteY = spriteDict["y"]
            if not isinstance(spriteY, int) or isinstance(spriteY, bool):
                raise invalidSceneException(f"Coordinata y dello sprite {spriteIndex} deve essere un intero.")
                
            flipH = spriteDict["flip_h"]
            if not isinstance(flipH, bool):
                raise invalidSceneException(f"flip_h dello sprite {spriteIndex} deve essere un booleano.")
                
            flipV = spriteDict["flip_v"]
            if not isinstance(flipV, bool):
                raise invalidSceneException(f"flip_v dello sprite {spriteIndex} deve essere un booleano.")
                
            rotation = spriteDict["rotation"]
            if not isinstance(rotation, int) or isinstance(rotation, bool):
                raise invalidSceneException(f"Rotazione dello sprite {spriteIndex} deve essere un intero.")
            if rotation not in {0, 90, 180, 270}:
                raise invalidSceneException(f"Rotazione dello sprite {spriteIndex} non valida (0, 90, 180, 270): {rotation}")
                
            # Mappa in chiavi camelCase per la compatibilità
            validatedSpritesList.append({
                "id": spriteId,
                "x": spriteX,
                "y": spriteY,
                "flipH": flipH,
                "flipV": flipV,
                "rotation": rotation
            })
            
        self.spritesList = validatedSpritesList
