OBJS= stddef.o

all: headers.o

headers.o: $(OBJS)
	$(LD) $^ -o $@ -r

.h.c:
	$(CC) -E $< -o $@

.c.o:
	$(LD) -b binary $< -o $@ -r

clean:
	-rm *.o

.PHONY: clean
