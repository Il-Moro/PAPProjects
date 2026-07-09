# Retro 2D Rendering Engine (Python)

**Studente:** Filippo Morello  
**Matricola:** SM3201475  

Questo progetto implementa la seconda parte dell'esame di Programmazione Avanzata e Parallela: un motore di rendering 2D in stile retro con palette indicizzata a 16 colori, tile map di sfondo e gestione degli sprite con trasparenza e trasformazioni spaziali (rotazione e flip).

## Requisiti di Sistema
Il progetto richiede Python 3 e le seguenti librerie di terze parti:
* `numpy` (per la manipolazione efficiente di matrici/frame buffer)
* `pillow` (usata esclusivamente per esportare l'immagine finale in formato PNG)

Puoi installarle tramite pip:
```bash
pip install numpy pillow
```

## Esecuzione del Renderer da Riga di Comando
Il programma può essere avviato specificando i 5 file richiesti nell'ordine definito:
```bash
python main.py <palette.json> <scene.json> <tiles.bin> <sprites.bin> <output.png>
```

### Esempio Pratico
Per renderizzare la scena di esempio inclusa nel progetto:
```bash
python main.py example/palette.json example/scene.json example/tiles.bin example/sprites.bin output.png
```
Questo genererà un file `output.png` che corrisponde esattamente al file di riferimento `example/reference.png`.

