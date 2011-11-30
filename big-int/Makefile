ESBMCDIR= $(shell pwd)/..

RM = rm -f
SRCS= bigint-test.cc bigint-func.cc bigint.cc
OBJ = bigint-test.o bigint-func.o bigint.o
PRG = test-bigint

all : $(PRG)

clean :
	$(RM) $(OBJ) $(PRG)
	-rm .deps

$(PRG) : $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(PRG) $(OBJ)

dep:

include $(ESBMCDIR)/common
