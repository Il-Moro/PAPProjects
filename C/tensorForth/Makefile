TARGET = tensorforth
CC = gcc
CFLAGS = -Wall -Wextra -g -fopenmp -march=native -I./tensor -I./stack -I./parsing
LIB = ./LIB

# Sorgenti per cartella
TENSOR_SRCS = tensor/tensor.c tensor/tensor_ops.c tensor/tensor_io.c
STACK_SRCS = stack/stack.c
PARSE_SRCS = parsing/token.c

LIB_SRCS = $(TENSOR_SRCS) $(STACK_SRCS) $(PARSE_SRCS)
LIB_OBJS = $(LIB_SRCS:.c=.o)

all: $(TARGET)

$(TARGET): tensorforth.o $(LIB)/libts.a
	$(CC) $(CFLAGS) -o $@ tensorforth.o -L$(LIB) -lts

$(LIB)/libts.a: $(LIB_OBJS)
	mkdir -p $(LIB)
	ar r $@ $^

# Regola generica: compila ogni .c nella sua stessa cartella
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

tensorforth.o: tensorforth.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(TARGET) tensorforth.o $(LIB_OBJS)
	rm -rf $(LIB)
