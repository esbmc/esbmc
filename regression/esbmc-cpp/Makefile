# Directories
algorithm := esbmc-algorithm
cpp := esbmc-cpp
inheritance := esbmc-inheritance
stream := esbmc-stream
string := esbmc-string
trycatch := esbmc-try_catch
vector := esbmc-vector
deque := esbmc-deque

# Clean
ifneq ($(filter clean,$(MAKECMDGOALS)),)
        TARGET := clean
endif
clean: all

# Make them all
all: $(algorithm) $(cpp) $(inheritance) $(stream) $(string) $(trycatch) $(vector) $(deque)

$(algorithm) $(cpp) $(inheritance) $(stream) $(string) $(trycatch) $(vector) $(deque):
	$(MAKE) --directory=$@ $(TARGET)

# Make each directory
algorithm: $(algorithm)
cpp: $(cpp)
inheritance: $(inheritance)
stream: $(stream)
string: $(string)
trycatch: $(trycatch)
vector: $(vector)
deque: $(deque)

# Allow parallel make
.PHONY: all $(algorithm) $(cpp) $(inheritance) $(stream) $(string) $(trycatch) $(vector) $(deque)
