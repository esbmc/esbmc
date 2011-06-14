OBJS= stddef.o

all: headers.o

headers.o: $(OBJS)
	$(LD) $^ -o $@ -r

.h.o:
	$(CC) -E $< -o $<.i
	$(LD) -b binary $<.i -o $@ -r

clean:
	-rm *.o
	-rm *.i

.PHONY: clean
