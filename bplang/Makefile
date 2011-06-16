SRC = bp_language.cpp bp_parser.cpp bp_typecheck.cpp expr2bp.cpp \
      y.tab.cpp lex.yy.cpp bp_parse_tree.cpp bp_typecheck_code.cpp \
      bp_typecheck_expr.cpp bp_util.cpp
OBJ = $(SRC:.cpp=.o)

INCLUDES= -I .. -I ../util

all: bplang.o

include ../config.inc
include ../common

###############################################################################

bplang.o: $(OBJ)
	$(LD) $(LDFLAGS) -r -o bplang.o $(OBJ)

y.tab.cpp: parser.y
	$(YACC) $(YFLAGS) $$flags -pyybp -d parser.y -o y.tab.cpp

y.tab.h: y.tab.cpp
	if [ -e y.tab.hpp ] ; then mv y.tab.hpp y.tab.h ; else \
        mv y.tab.cpp.h y.tab.h ; fi

lex.yy.cpp: scanner.l
	$(LEX) -Pyybp -olex.yy.cpp scanner.l

# extra dependencies
y.tab.o: y.tab.cpp y.tab.h
lex.yy.o: y.tab.cpp lex.yy.cpp y.tab.h

clean:
	rm -f $(OBJ) *.o y.tab.h y.tab.cpp lex.yy.cpp y.tab.cpp.output \
              y.output
	-rm *.d

dep: y.tab.cpp y.tab.h lex.yy.cpp
