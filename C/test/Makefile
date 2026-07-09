CC     = gcc
CFLAGS = -Wall -Wextra -g -fopenmp \
         -I../tensor \
         -I../stack \
         -I../parsing
LIB    = ../LIB

TESTS = tensorTest stackTest tokenTest ioTest time

all: $(TESTS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< -L$(LIB) -lts -lm

clean:
	rm -f $(TESTS) *.o