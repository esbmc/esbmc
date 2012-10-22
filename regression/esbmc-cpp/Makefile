# Directories
algorithm := esbmc-algorithm
cpp := esbmc-cpp
inheritance := esbmc-inheritance
stream := esbmc-stream
string := esbmc-string
trycatch := esbmc-try_catch
vector := esbmc-vector
deque := esbmc-deque
list := esbmc-list
stack := esbmc-stack

# Clean
ifneq ($(filter clean,$(MAKECMDGOALS)),)
        TARGET := clean
endif
clean: all

# llbmc
ifneq ($(filter llvm,$(MAKECMDGOALS)),)
        TARGET := llvm
endif
llvm: all

# Make them all
all: $(algorithm) $(cpp) $(inheritance) $(stream) $(string) $(trycatch) $(vector) $(deque) $(list) $(stack)

$(algorithm) $(cpp) $(inheritance) $(stream) $(string) $(trycatch) $(vector) $(deque) $(list) $(stack):
	$(MAKE) --directory=$@ $(TARGET)

#Make only STL directories
stl: $(algorithm) $(vector) $(deque) $(list) $(stack)

# Make each directory
algorithm: $(algorithm)
cpp: $(cpp)
inheritance: $(inheritance)
stream: $(stream)
string: $(string)
trycatch: $(trycatch)
vector: $(vector)
deque: $(deque)
list: $(list)
stack: $(stack)

# Allow parallel make
.PHONY: all $(algorithm) $(cpp) $(inheritance) $(stream) $(string) $(trycatch) $(vector) $(deque) $(list) $(stack)
