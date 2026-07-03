// Nome: Filippo
// Cognome: Morello
// Matricola: SM3201475

#include <stdio.h>
#include <stdlib.h> 

#include "stack.h"
#include "tensor.h"

int main(int argc, char *argv[]) {
	if (argc > 1) {
    	FILE *fd = fopen(argv[1], "r");
    
    	if(fd == NULL){
			perror("Impossibile aprire file");
    	}
		
	} else{
		perror("File richisto in inputo");
		return -1;
	}

  	return 0;
}

