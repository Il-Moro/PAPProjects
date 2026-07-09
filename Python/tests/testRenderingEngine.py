# Nome: Filippo
# Cognome: Morello
# Matricola: SM3201475

import os
import sys
import unittest
import numpy as np
from PIL import Image

# Aggiunge la cartella contenitrice (Python/) a sys.path per consentire l'avvio diretto del test
parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parentDir not in sys.path:
    sys.path.insert(0, parentDir)

from renderingEngine.palette import Palette
from renderingEngine.virtualVram import VirtualVRAM
from renderingEngine.SceneParser import SceneParser
from renderingEngine.blitter import Blitter
from renderingEngine.RenderingPipeline import RenderingPipeline
from renderingEngine.customExceptions import (
    invalidPaletteException,
    invalidVramException,
    invalidSceneException,
    renderingException
)

class testRenderingEngine(unittest.TestCase):
    
    def setUp(self):
        testDir = os.path.dirname(os.path.abspath(__file__))
        self.exampleDir = os.path.join(os.path.dirname(testDir), "example")
        self.palettePath = os.path.join(self.exampleDir, "palette.json")
        self.scenePath = os.path.join(self.exampleDir, "scene.json")
        self.tilesPath = os.path.join(self.exampleDir, "tiles.bin")
        self.spritesPath = os.path.join(self.exampleDir, "sprites.bin")
        self.referencePath = os.path.join(self.exampleDir, "reference.png")
        self.outputPath = os.path.join(testDir, "example_output_test.png")
        
    def tearDown(self):
        if os.path.exists(self.outputPath):
            os.remove(self.outputPath)
            
    def testPaletteLoadingAndValidation(self):
        print("\n[TEST] testPaletteLoadingAndValidation...")
        # Caricamento corretto
        validPalette = Palette(self.palettePath)
        self.assertEqual(len(validPalette.colorsList), 16)
        
        # Risoluzione colore corretto
        rgbVal = validPalette.resolveColor(0)
        self.assertEqual(rgbVal, (255, 0, 255))
        
        # Indici fuori range
        with self.assertRaises(invalidPaletteException):
            validPalette.resolveColor(16)
        with self.assertRaises(invalidPaletteException):
            validPalette.resolveColor(-1)
            
        paletteInstance = Palette()
        
        # Lunghezza errata (15 elementi)
        with self.assertRaises(invalidPaletteException):
            paletteInstance.validateAndSetPalette([[0, 0, 0]] * 15)
            
        # Elemento non lista
        with self.assertRaises(invalidPaletteException):
            paletteInstance.validateAndSetPalette([[0, 0, 0]] * 15 + ["errore"])
            
        # Numero di canali errato (2 canali)
        with self.assertRaises(invalidPaletteException):
            paletteInstance.validateAndSetPalette([[0, 0, 0]] * 15 + [[0, 0]])
            
        # Valore non intero
        with self.assertRaises(invalidPaletteException):
            paletteInstance.validateAndSetPalette([[0, 0, 0]] * 15 + [[0, 0, 1.2]])
            
        # Valore fuori range 0-255
        with self.assertRaises(invalidPaletteException):
            paletteInstance.validateAndSetPalette([[0, 0, 0]] * 15 + [[0, 0, 256]])
        print("[TEST] testPaletteLoadingAndValidation superato!")
            
    def testVirtualVramLoadingAndDecoding(self):
        print("[TEST] testVirtualVramLoadingAndDecoding...")
        vramInstance = VirtualVRAM()
        
        # Caricamento corretto delle matrici
        vramInstance.loadTileSheet(self.tilesPath)
        vramInstance.loadSpriteSheet(self.spritesPath)
        
        self.assertEqual(vramInstance.tilesMatrix.shape, (256, 256))
        self.assertEqual(vramInstance.spritesMatrix.shape, (256, 256))
        
        # Errore se il file non esiste
        with self.assertRaises(invalidVramException):
            vramInstance.loadTileSheet("file_inesistente.bin")
            
        # Errore se la dimensione e' errata
        dummyFilePath = "dummy_test.bin"
        with open(dummyFilePath, "wb") as f:
            f.write(b"\x00" * 50)
        try:
            with self.assertRaises(invalidVramException):
                vramInstance.loadTileSheet(dummyFilePath)
        finally:
            if os.path.exists(dummyFilePath):
                os.remove(dummyFilePath)
        print("[TEST] testVirtualVramLoadingAndDecoding superato!")
                
    def testSceneParsingAndValidation(self):
        print("[TEST] testSceneParsingAndValidation...")
        parserInstance = SceneParser(self.scenePath)
        self.assertEqual(parserInstance.transparentIndex, 0)
        self.assertEqual(parserInstance.tileMap.shape, (15, 20))
        self.assertTrue(len(parserInstance.spritesList) > 0)
        
        testParser = SceneParser()
        
        # JSON con chiavi mancanti
        missingKeys = {"transparent_index": 0, "tile_map": [[0]*20]*15}
        with self.assertRaises(invalidSceneException):
            testParser.validateAndSetScene(missingKeys)
            
        # Indice trasparenza non valido
        invalidTrans = {"transparent_index": 16, "tile_map": [[0]*20]*15, "sprites": []}
        with self.assertRaises(invalidSceneException):
            testParser.validateAndSetScene(invalidTrans)
            
        # Righe tile map errate (14 righe)
        invalidRows = {"transparent_index": 0, "tile_map": [[0]*20]*14, "sprites": []}
        with self.assertRaises(invalidSceneException):
            testParser.validateAndSetScene(invalidRows)
            
        # Colonne tile map errate (19 colonne)
        invalidCols = {"transparent_index": 0, "tile_map": [[0]*19]*15, "sprites": []}
        with self.assertRaises(invalidSceneException):
            testParser.validateAndSetScene(invalidCols)
            
        # ID tile non valido (> 63)
        invalidId = {"transparent_index": 0, "tile_map": [[0]*19 + [64]] * 15, "sprites": []}
        with self.assertRaises(invalidSceneException):
            testParser.validateAndSetScene(invalidId)
            
        # Rotazione dello sprite errata
        invalidRot = {
            "transparent_index": 0,
            "tile_map": [[0]*20]*15,
            "sprites": [{"id": 0, "x": 0, "y": 0, "flip_h": False, "flip_v": False, "rotation": 45}]
        }
        with self.assertRaises(invalidSceneException):
            testParser.validateAndSetScene(invalidRot)
        print("[TEST] testSceneParsingAndValidation superato!")
            
    def testBlitterTransformations(self):
        print("[TEST] testBlitterTransformations...")
        blitterInstance = Blitter()
        frameBuffer = np.zeros((480, 640), dtype=np.uint8)
        spriteSheet = np.zeros((256, 256), dtype=np.uint8)
        
        spriteSheet[0, :4] = [1, 2, 3, 4]
        
        # Blit standard
        blitterInstance.blitSprite(frameBuffer, 0, 0, 0, False, False, 0, spriteSheet, 0)
        self.assertEqual(frameBuffer[0, 0], 1)
        self.assertEqual(frameBuffer[0, 1], 2)
        
        # Blit con flip orizzontale
        frameBuffer.fill(0)
        blitterInstance.blitSprite(frameBuffer, 0, 0, 0, True, False, 0, spriteSheet, 0)
        self.assertEqual(frameBuffer[0, 63], 1)
        self.assertEqual(frameBuffer[0, 62], 2)
        
        # Blit parzialmente fuori schermo (clipping)
        spriteSheet[0, 61] = 9
        frameBuffer.fill(0)
        blitterInstance.blitSprite(frameBuffer, 0, -60, 0, False, False, 0, spriteSheet, 0)
        self.assertEqual(frameBuffer[0, 1], 9)
        print("[TEST] testBlitterTransformations superato!")
        
    def testRenderingPipelineIntegration(self):
        print("[TEST] testRenderingPipelineIntegration...")
        pipelineInstance = RenderingPipeline()
        
        # Avvia rendering
        pipelineInstance.render(
            self.palettePath,
            self.scenePath,
            self.tilesPath,
            self.spritesPath,
            self.outputPath
        )
        
        self.assertTrue(os.path.exists(self.outputPath))
        
        genImg = Image.open(self.outputPath)
        refImg = Image.open(self.referencePath)
        
        genArr = np.array(genImg)
        refArr = np.array(refImg)
        
        diff = np.sum(genArr != refArr)
        self.assertEqual(diff, 0, f"Rendering non corrispondente! Pixel diversi: {diff}")
        print("[TEST] testRenderingPipelineIntegration superato!")

if __name__ == "__main__":
    print("=== AVVIO SUITE COMPLETA TEST RENDERING ENGINE ===")
    unittest.main(verbosity=2)
