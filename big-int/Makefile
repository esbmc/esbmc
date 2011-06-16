include ../config.inc
include ../common

RM = rm -f
OBJ = bigint-test.o bigint-func.o bigint.o
PRG = test-bigint

all : $(PRG)

clean :
	$(RM) $(OBJ) $(PRG)
	-rm *.d

$(PRG) : $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(PRG) $(OBJ)

dep:
