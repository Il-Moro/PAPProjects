CC     = gcc
CFLAGS = -Wall -Wextra -g -fopenmp \
         -I../../tensor \
         -I../../stack \
         -I../../parsing
LIB    = ../../LIB

TARGETS = bench_arithmetic bench_comparison bench_logical bench_selection bench_matrix bench_tensor_ops bench_other_ops

all: $(TARGETS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< -L$(LIB) -lts -lm

clean:
	rm -f $(TARGETS) *.o
