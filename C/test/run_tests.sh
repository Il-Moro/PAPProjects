#!/bin/bash

# --- COLORI PER OUTPUT ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # Nessun colore

echo -e "${YELLOW}=== Avvio Suite di Test TensorForth ===${NC}"

# 1. Compilazione del programma principale
echo "Compilazione dell'interprete..."
cd "$(dirname "$0")/../tensorForth" || exit 1
make clean > /dev/null 2>&1
make
if [ $? -ne 0 ]; then
    echo -e "${RED}Errore di compilazione dell'interprete!${NC}"
    exit 1
fi
cd ..

# 2. Compilazione dei test unitari C (se pronti e non vuoti)
echo "Compilazione dei test unitari C..."
cd test || exit 1
make clean > /dev/null 2>&1
make > /dev/null 2>&1
make_tests_status=$?
cd ..

# Se i test unitari C compilano, li eseguiamo
if [ $make_tests_status -eq 0 ]; then
    echo -e "${YELLOW}Esecuzione dei test unitari C:${NC}"
    for test_bin in test/stackTest test/tensorTest test/tensorforthTest test/tokenTest; do
        if [ -f "$test_bin" ]; then
            echo -n "Esecuzione di $test_bin... "
            ./"$test_bin" > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}OK${NC}"
            else
                echo -e "${RED}FALLITO${NC}"
            fi
        fi
    done
else
    echo -e "${YELLOW}Test unitari C vuoti o non pronti per la compilazione (saltati)${NC}"
fi

# Crea una cartella per eventuali output generati
mkdir -p examples/output

# Se viene passato un argomento specifico, esegue solo quel test senza silenziare l'output
if [ $# -gt 0 ]; then
    test_file="$1"
    # Se il percorso non inizia con "examples/", lo aggiunge per comodità se esiste lì
    if [[ ! "$test_file" =~ ^examples/ ]] && [ -f "examples/$test_file" ]; then
        test_file="examples/$test_file"
    fi
    
    if [ ! -f "$test_file" ]; then
        echo -e "${RED}Errore: Il file '$test_file' non esiste.${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Esecuzione di $test_file in corso...${NC}"
    ./tensorForth/TensorForth "$test_file"
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}Esecuzione completata con successo! (Codice d'uscita: 0)${NC}"
        exit 0
    else
        echo -e "${RED}Esecuzione fallita! (Codice d'uscita: $exit_code)${NC}"
        exit $exit_code
    fi
fi

# Elenco di default se non viene passato alcun parametro
EXAMPLES=(
    "duplicate.tensorforth"
    "random_matmul.tensorforth"
    "image_blur.tensorforth"
    "detect_edges.tensorforth"
    "convert_to_bw.tensorforth"
    "save_tensor.tensorforth"
    "game_of_life.tensorforth"
)

failed=0
passed=0

echo -e "\nEsecuzione dei programmi di esempio:"
for ex in "${EXAMPLES[@]}"; do
    echo -n "Test di examples/$ex... "
    
    # Esegue il programma silenziosamente
    ./tensorForth/TensorForth "examples/$ex" > /dev/null 2>&1
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}OK${NC}"
        passed=$((passed + 1))
    else
        echo -e "${RED}FALLITO (Codice d'uscita: $exit_code)${NC}"
        failed=$((failed + 1))
    fi
done

echo -e "\n--- Riepilogo dei Test ---"
echo -e "Passati: ${GREEN}$passed${NC}"
echo -e "Falliti: ${RED}$failed${NC}"

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}Tutti i test sono stati completati con successo!${NC}"
    exit 0
else
    echo -e "${RED}Ci sono stati errori nei test.${NC}"
    exit 1
fi
