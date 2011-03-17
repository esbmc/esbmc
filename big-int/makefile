CXX = c++
CXXFLAGS = -O
RM = rm -f
OBJ = bigint-test.o bigint-func.o bigint.o
PRG = test-bigint

all : $(PRG)

clean :
	$(RM) $(OBJ) $(PRG)

$(PRG) : $(OBJ)
	$(CXX) -o $(PRG) $(OBJ)

dep:
