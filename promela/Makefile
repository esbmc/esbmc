#SRC = promela_language.cpp promela_parser.cpp promela_typecheck.cpp \
#      expr2promela.cpp y.tab.cpp lex.yy.cpp promela_parse_tree.cpp
SRC = expr2promela.cpp promela_parse_tree.cpp ../langapi/promela_dummy.cpp
OBJ = $(SRC:.cpp=.o)

INCLUDES= -I .. -I ../util

all: promela.o

include ../config.inc
include ../common

###############################################################################

promela.o: $(OBJ)
	$(LD) -r -o promela.o $(OBJ)

y.tab.cpp: parser.y
	$(YACC) $(YFLAGS) $$flags -pyypromela -d parser.y -o y.tab.cpp

y.tab.h: y.tab.cpp
	if [ -e y.tab.hpp ] ; then mv y.tab.hpp y.tab.h ; else \
        mv y.tab.cpp.h y.tab.h ; fi

#lex.yy.cpp: scanner.l
#	$(LEX) -Pyypromela -olex.yy.cpp scanner.l

# extra dependencies
y.tab.o: y.tab.cpp y.tab.h
lex.yy.o: y.tab.cpp lex.yy.cpp y.tab.h

clean:
	rm -f $(OBJ) *.o y.tab.h y.tab.cpp lex.yy.cpp y.tab.cpp.output

# dep: y.tab.cpp y.tab.h lex.yy.cpp
