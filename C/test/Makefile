CC = gcc
CFLAGS = -Wall -Wextra -g -fopenmp -I../tensorForth
LIB = ../tensorForth/LIB

TESTS = stackTest tensorTest tensorforthTest tokenTest

all: $(TESTS)

%.o: %.c $(LIB)/libts.a
	$(CC) $(CFLAGS) -o $@ $< -L$(LIB) -lts -lm

clean:
	rm -f $(TESTS) *.o