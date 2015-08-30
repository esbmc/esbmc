#Note: you have to install the goto-cc tool to run this Makefile 
CC=goto-cc
CFLAGS=

all: *.c
	-make $(patsubst %.c, %, $(wildcard *.c))

clean:
	rm -f $(patsubst %.c, %, $(wildcard *.c))
	rm -f *.tmp *.*~
