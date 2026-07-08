CC     = gcc
CFLAGS = -Wall -Wextra -g -fopenmp \
         -I../tensorForth/tensor \
         -I../tensorForth/stack \
         -I../tensorForth/parsing
LIB    = ../tensorForth/LIB

TESTS = tensorTest stackTest tokenTest ioTest time

all: $(TESTS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< -L$(LIB) -lts -lm

clean:
	rm -f $(TESTS) *.o