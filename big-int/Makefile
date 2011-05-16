include ../config.inc
include ../config.inc

RM = rm -f
OBJ = bigint-test.o bigint-func.o bigint.o
PRG = test-bigint

all : $(PRG)

clean :
	$(RM) $(OBJ) $(PRG)

$(PRG) : $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(PRG) $(OBJ)

dep:
