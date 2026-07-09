# Nome: Filippo
# Cognome: Morello
# Matricola: SM3201475

import os
import sys

# Aggiunge la cartella superiore a sys.path per permettere l'avvio diretto di main.py
parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parentDir not in sys.path:
    sys.path.insert(0, parentDir)

from renderingEngine.pipeline.RenderingPipeline import RenderingPipeline
from renderingEngine.exceptions.customExceptions import renderingEngineException

def main():
    # Verifica che siano stati passati esattamente 5 argomenti
    if len(sys.argv) != 6:
        print(
            "Errore: Numero di argomenti non corretto.\n"
            "Uso: python main.py <palette.json> <scene.json> <tiles.bin> <sprites.bin> <output.png>",
            file=sys.stderr
        )
        sys.exit(1)
        
    palettePath = sys.argv[1]
    scenePath = sys.argv[2]
    tilesPath = sys.argv[3]
    spritesPath = sys.argv[4]
    outputPath = sys.argv[5]
    
    try:
        # Avvia il processo di composizione e salvataggio
        pipelineInstance = RenderingPipeline()
        pipelineInstance.render(
            palettePath=palettePath,
            scenePath=scenePath,
            tilesPath=tilesPath,
            spritesPath=spritesPath,
            outputPath=outputPath
        )
        print(f"Rendering completato con successo: {outputPath}")
        sys.exit(0)
        
    except renderingEngineException as expectedException:
        print(f"Errore di configurazione o esecuzione: {expectedException}", file=sys.stderr)
        sys.exit(1)
        
    except Exception as unexpectedException:
        print(f"Errore generico imprevisto: {unexpectedException}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
