CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
LDLIBS  ?= -lm

all: matops

matops: src/main.c src/matops.c src/matops.h
	$(CC) $(CFLAGS) -o $@ src/main.c src/matops.c $(LDLIBS)

mat_tests: tests/mat_tests.c src/matops.c src/matops.h
	$(CC) $(CFLAGS) -Isrc -o mat_tests tests/mat_tests.c src/matops.c $(LDLIBS)

test: mat_tests matops
	./mat_tests
	./matops bench -n 160 > /dev/null && echo "bench OK"
	./matops solve -n 100 && ./matops det -n 60 > /dev/null

clean:
	rm -f matops mat_tests

.PHONY: all test clean
