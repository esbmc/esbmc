CC=	./goto-cc

OBJS= ctype.o getopt.o io.o math.o pthread_lib.o stdlib.o string.o time.o

clibrary.o: $(OBJS)
	$(CC) $(OBJS) -o $@ -c

clean:
	-rm *.o

.PHONY: clean
