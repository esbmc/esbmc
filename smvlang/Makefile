SRC = smv_language.cpp smv_parser.cpp smv_typecheck.cpp expr2smv.cpp \
      y.tab.cpp lex.yy.cpp smv_parse_tree.cpp
OBJ = $(SRC:.cpp=.o)

INCLUDES= -I .. -I ../util

all: smvlang.o

include ../config.inc
include ../common

###############################################################################

smvlang.o: $(OBJ)
	$(LD) $(LDFLAGS) -r -o smvlang.o $(OBJ)

y.tab.cpp: parser.y
	$(YACC) $(YFLAGS) $$flags -pyysmv -d parser.y -o y.tab.cpp

y.tab.h: y.tab.cpp
	if [ -e y.tab.hpp ] ; then mv y.tab.hpp y.tab.h ; else \
        mv y.tab.cpp.h y.tab.h ; fi

lex.yy.cpp: scanner.l
	$(LEX) -Pyysmv -olex.yy.cpp scanner.l

# extra dependencies
y.tab.o: y.tab.cpp y.tab.h
lex.yy.o: y.tab.cpp lex.yy.cpp y.tab.h

clean:
	rm -f $(OBJ) *.o y.tab.h y.tab.cpp lex.yy.cpp y.tab.cpp.output \
              y.output
	-rm *.d

dep: y.tab.cpp y.tab.h lex.yy.cpp
