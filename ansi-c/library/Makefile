CC=	gcc
SRCS= ctype.i getopt.i io.i math.i pthread_lib.i stdlib.i string.i time.i

all: clibrary
.SUFFIXES: .i

.c.i:
	$(CC) $(CFLAGS) -E $< -o $@

FILES= $(SRCS:.c=.i)

clibrary: $(FILES)

clean:
	-rm *.i

.PHONY: clean clibrary
