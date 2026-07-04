TARGET = tensorforth
CC = gcc
CFLAGS = -Wall -Wextra -g -fopenmp -march=native
LIB = ./LIB

all: $(TARGET)

$(TARGET): tensorforth.o $(LIB)/libts.a
	$(CC) $(CFLAGS) -o $@ tensorforth.o -L$(LIB) -lts

$(LIB)/libts.a: stack.o tensor.o 
	mkdir -p $(LIB)
	ar r $@ $^

%.o: %.c $(wildcard *.h)
	$(CC) $(CFLAGS) -c $<

.PHONY: clean
clean:
	rm -f $(TARGET) *.o
	rm -rf $(LIB)
