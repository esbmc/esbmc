SRC = cvc_language.cpp cvc_typecheck.cpp expr2cvc.cpp cvc_parser.cpp \
      y.tab.cpp lex.yy.cpp
OBJ = cvc_language.o cvc_typecheck.o expr2cvc.o cvc_parser.o \
      y.tab.o lex.yy.o

INCLUDES= -I .. -I ../util

all: cvclang.o

include ../config.inc
include ../common

###############################################################################

cvclang.o: $(OBJ)
	$(LD) $(LDFLAGS) -r -o cvclang.o $(OBJ)

y.tab.cpp: parser.y
	$(YACC) $(YFLAGS) $$flags -pyycvc -d parser.y -o y.tab.cpp

y.tab.h: y.tab.cpp
	if [ -e y.tab.hpp ] ; then mv y.tab.hpp y.tab.h ; else \
        mv y.tab.cpp.h y.tab.h ; fi

lex.yy.cpp: scanner.l
	$(LEX) -Pyycvc -olex.yy.cpp scanner.l

# extra dependencies
y.tab.o: y.tab.cpp y.tab.h
lex.yy.o: y.tab.cpp lex.yy.cpp y.tab.h

clean:
	rm -f $(OBJ) *.o y.tab.h y.tab.cpp lex.yy.cpp y.output
	-rm *.d
dep: y.tab.cpp y.tab.h lex.yy.cpp
