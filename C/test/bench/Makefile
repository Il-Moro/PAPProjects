CC = gcc
CFLAGS = -Wall -Wextra -g -fopenmp -I../../tensorForth
LIB = ../../tensorForth/LIB

TARGETS = bench_arithmetic bench_comparison bench_logical bench_selection

all: $(TARGETS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< -L$(LIB) -lts -lm

clean:
	rm -f $(TARGETS) *.o
