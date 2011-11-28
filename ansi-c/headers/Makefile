OBJS= stddef.o stdarg.o stdbool.o

all: headers.o

headers.o: $(OBJS)
	$(LD) $^ -o $@ -r

.h.o:
	$(LD) -b binary $< -o $@ -r

clean:
	-rm *.o
	-rm *.d

.PHONY: clean
