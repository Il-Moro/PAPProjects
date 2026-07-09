# Nome: Filippo
# Cognome: Morello
# Matricola: SM3201475

import numpy as np
from PIL import Image
from renderingEngine.palette import Palette
from renderingEngine.virtualVram import VirtualVRAM
from renderingEngine.SceneParser import SceneParser
from renderingEngine.blitter import Blitter
from renderingEngine.customExceptions import renderingException, renderingEngineException

# Pipeline principale di coordinamento del rendering
class RenderingPipeline:
    
    def render(self, palettePath, scenePath, tilesPath, spritesPath, outputPath):
        try:
            # 1. Carica e valida gli asset
            paletteManager = Palette(palettePath)
            vramManager = VirtualVRAM()
            vramManager.loadTileSheet(tilesPath)
            vramManager.loadSpriteSheet(spritesPath)
            
            parsedScene = SceneParser(scenePath)
            
            # 2. Inizializza il frame buffer vuoto (640x480)
            frameBuffer = np.zeros((480, 640), dtype=np.uint8)
            blittingEngine = Blitter()
            
            # 3. Blitting della tile map di sfondo
            blittingEngine.blitTileMap(
                frameBuffer=frameBuffer,
                tileMap=parsedScene.tileMap,
                tileSheet=vramManager.tilesMatrix
            )
            
            # 4. Blitting dei singoli sprite
            for spriteInfo in parsedScene.spritesList:
                blittingEngine.blitSprite(
                    frameBuffer=frameBuffer,
                    spriteId=spriteInfo["id"],
                    x=spriteInfo["x"],
                    y=spriteInfo["y"],
                    flipH=spriteInfo["flipH"],
                    flipV=spriteInfo["flipV"],
                    rotation=spriteInfo["rotation"],
                    spriteSheet=vramManager.spritesMatrix,
                    transparentIndex=parsedScene.transparentIndex
                )
                
            # 5. Converte il frame buffer indicizzato in RGB mappando i colori
            rgbFrameBuffer = np.zeros((480, 640, 3), dtype=np.uint8)
            for colorIndex in range(16):
                rgbFrameBuffer[frameBuffer == colorIndex] = paletteManager.resolveColor(colorIndex)
                
            # 6. Salva il file immagine PNG risultante
            try:
                outputImage = Image.fromarray(rgbFrameBuffer, mode="RGB")
                outputImage.save(outputPath, format="PNG")
            except Exception as fileSaveError:
                raise renderingException(f"Errore nel salvataggio dell'immagine finale: {fileSaveError}")
                
        except renderingEngineException as expectedError:
            raise expectedError
        except Exception as unexpectedError:
            raise renderingException(f"Errore imprevisto durante il rendering: {unexpectedError}")
