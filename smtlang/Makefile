SRC = lex.yysmt.cpp smt_parse_tree.cpp smt_parser.cpp \
      smt_language.cpp smt_typecheck.cpp smt_link.cpp expr2smt.cpp \
      smt_typecheck_expr.cpp smt_logics.cpp smt_finalizer.cpp \
      builtin_theories.cpp smt_finalizer_generic.cpp \
      smt_finalizer_AUFLIA.cpp smt_finalizer_AUFLIRA.cpp \
      smt_finalizer_AUFNIRA.cpp smt_finalizer_QF_IDL.cpp \
      smt_finalizer_QF_LIA.cpp smt_finalizer_QF_LRA.cpp \
      smt_finalizer_QF_UFBV32.cpp smt_strings.cpp

OBJ1 = $(SRC:.c=.o)
OBJ = $(OBJ1:.cpp=.o)

INCLUDES= -I .. -I ../util

all: smtlang.o

include ../config.inc
include ../common

###############################################################################

lex.yysmt.cpp: scanner.l
	$(LEX) -Pyysmt -olex.yysmt.cpp -i scanner.l

smtlang.o: $(OBJ)
	$(LD) -r -o smtlang.o $(OBJ)

smt.tab.cpp: parser.y
	$(YACC) $(YFLAGS) $$flags -y -pyysmt -d parser.y -o smt.tab.cpp

smt.tab.h: smt.tab.cpp
	if [ -e smt.tab.hpp ] ; then mv smt.tab.hpp smt.tab.h ; else \
	if [ -e smt.tab.cpp.h ] ; then mv smt.tab.cpp.h smt.tab.h ; fi ; fi

clean:
	rm -f $(OBJ) *.o *.d lex.yysmt.cpp smt.tab.cpp smt.tab.h smt.tab.hpp

dep: lex.yysmt.cpp smt.tab.cpp smt.tab.h smt.tab.hpp
