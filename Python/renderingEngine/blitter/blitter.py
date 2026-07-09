# Nome: Filippo
# Cognome: Morello
# Matricola: SM3201475

import numpy as np
from renderingEngine.exceptions.customExceptions import renderingException

# Blitter per copiare tile e sprite trasformati sul frame buffer
class Blitter:
    
    def blitTileMap(self, frameBuffer, tileMap, tileSheet):
        try:
            # Controllo dimensioni delle componenti
            if frameBuffer.shape != (480, 640):
                raise renderingException(f"Frame buffer non corretto: {frameBuffer.shape}")
            if tileMap.shape != (15, 20):
                raise renderingException(f"Tile map non corretta: {tileMap.shape}")
            if tileSheet.shape != (256, 256):
                raise renderingException(f"Tile sheet non corretto: {tileSheet.shape}")
                
            # Disegna la mappa del fondale riga per riga
            for rowIndex in range(15):
                for colIndex in range(20):
                    tileId = int(tileMap[rowIndex, colIndex])
                    
                    # Identifica riga e colonna del tile (dimensione 32x32) nel tile sheet (8x8 tile)
                    tileSheetRow = tileId // 8
                    tileSheetCol = tileId % 8
                    
                    tilePixels = tileSheet[
                        tileSheetRow * 32 : (tileSheetRow + 1) * 32,
                        tileSheetCol * 32 : (tileSheetCol + 1) * 32
                    ]
                    
                    # Copia nella posizione corretta del frame buffer
                    frameBuffer[
                        rowIndex * 32 : (rowIndex + 1) * 32,
                        colIndex * 32 : (colIndex + 1) * 32
                    ] = tilePixels
                    
        except Exception as errorObject:
            if isinstance(errorObject, renderingException):
                raise errorObject
            raise renderingException(f"Errore nel disegno del fondale: {errorObject}")
            
    def blitSprite(
        self,
        frameBuffer,
        spriteId,
        x,
        y,
        flipH,
        flipV,
        rotation,
        spriteSheet,
        transparentIndex
    ):
        try:
            if frameBuffer.shape != (480, 640):
                raise renderingException(f"Frame buffer non corretto: {frameBuffer.shape}")
            if spriteSheet.shape != (256, 256):
                raise renderingException(f"Sprite sheet non corretto: {spriteSheet.shape}")
            if spriteId < 0 or spriteId > 15:
                raise renderingException(f"Id sprite non valido: {spriteId}")
            if rotation not in {0, 90, 180, 270}:
                raise renderingException(f"Rotazione non valida: {rotation}")
                
            # Trova riga e colonna dello sprite nel foglio 4x4
            spriteSheetRow = spriteId // 4
            spriteSheetCol = spriteId % 4
            
            # Copia locale dello sprite di base (64x64) per applicare le trasformazioni
            spritePixels = spriteSheet[
                spriteSheetRow * 64 : (spriteSheetRow + 1) * 64,
                spriteSheetCol * 64 : (spriteSheetCol + 1) * 64
            ].copy()
            
            # Applica flip orizzontale o verticale
            if flipH:
                spritePixels = np.fliplr(spritePixels)
            if flipV:
                spritePixels = np.flipud(spritePixels)
                
            # Applica rotazione counter-clockwise
            if rotation == 90:
                spritePixels = np.rot90(spritePixels, 1)
            elif rotation == 180:
                spritePixels = np.rot90(spritePixels, 2)
            elif rotation == 270:
                spritePixels = np.rot90(spritePixels, 3)
                
            frameBufferHeight, frameBufferWidth = frameBuffer.shape
            
            # Calcola l'area visibile sul frame buffer gestendo il clipping (sprite parzialmente fuori)
            targetYStart = max(0, y)
            targetYEnd = min(frameBufferHeight, y + 64)
            targetXStart = max(0, x)
            targetXEnd = min(frameBufferWidth, x + 64)
            
            # Se è totalmente fuori schermo non disegna
            if targetYStart >= targetYEnd or targetXStart >= targetXEnd:
                return
                
            # Calcola l'indice iniziale e finale all'interno della matrice dello sprite
            spriteYStart = targetYStart - y
            spriteYEnd = spriteYStart + (targetYEnd - targetYStart)
            spriteXStart = targetXStart - x
            spriteXEnd = spriteXStart + (targetXEnd - targetXStart)
            
            spriteSlice = spritePixels[spriteYStart:spriteYEnd, spriteXStart:spriteXEnd]
            frameBufferSlice = frameBuffer[targetYStart:targetYEnd, targetXStart:targetXEnd]
            
            # Copia i pixel non trasparenti
            nonTransparentMask = (spriteSlice != transparentIndex)
            frameBufferSlice[nonTransparentMask] = spriteSlice[nonTransparentMask]
            
        except Exception as errorObject:
            if isinstance(errorObject, renderingException):
                raise errorObject
            raise renderingException(f"Errore nel disegno dello sprite: {errorObject}")
