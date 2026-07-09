# TensorForth - Interprete Stack-Based per Tensori (C)

**Studente:** Filippo Morello  
**Matricola:** SM3201475  

Questo progetto implementa la prima parte dell'esame di Programmazione Avanzata e Parallela: un interprete interamente scritto in C per un linguaggio stack-based orientato ai tensori (TensorForth), parallelizzato con OpenMP e ottimizzato tramite istruzioni intrinseche AVX2.

## Requisiti di Sistema
* Sistema Operativo: Linux (es. Ubuntu)
* Compilatore: `gcc`
* Supporto per **OpenMP** e vettorizzazione **AVX2** (incluso di default nelle CPU moderne)

## Struttura della Directory C
* `parsing/`, `stack/`, `tensor/`: contengono i sorgenti dell'interprete.
* `examples/`: programmi d'esempio in formato `.tensorforth`.

## Compilazione
Per compilare l'eseguibile `tensorforth`, posizionati nella cartella `C/` ed esegui `make`:
```bash
make
```
Questo genererà l'eseguibile `tensorforth` nella stessa cartella.

## Esecuzione di un Programma
Per eseguire un file TensorForth:
```bash
./tensorforth <percorso_sorgente.tensorforth>
```
Ad esempio:
```bash
./tensorforth examples/image_blur.tensorforth
```

## Esecuzione degli Esempi
Per verificare il funzionamento dell'interprete, puoi eseguire uno dei programmi di esempio presenti nella cartella `examples/` come indicato sopra. Gli output generati (es. immagini PGM o file tensore binari) verranno salvati all'interno della cartella `examples/output/` (la cartella verrà creata automaticamente se non esiste).

## Pulizia dei file compilati
Per ripulire i file temporanei `.o`, `.a` e l'eseguibile prima della consegna:
```bash
make clean
```
