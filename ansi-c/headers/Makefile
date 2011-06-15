OBJS= stddef.o stdarg.o

all: headers.o

headers.o: $(OBJS)
	$(LD) $^ -o $@ -r

.h.o:
	$(LD) -b binary $< -o $@ -r

clean:
	-rm *.o

.PHONY: clean
