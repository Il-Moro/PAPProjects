CC = gcc
CFLAGS = -Wall -Wextra -g -fopenmp -I../tensorForth
LIB = ../tensorForth/LIB

TESTS = tensorTest time

all: $(TESTS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< -L$(LIB) -lts -lm

clean:
	rm -f $(TESTS) *.o