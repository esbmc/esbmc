# Directories
algorithm := esbmc-algorithm
cpp := esbmc-cpp
inheritance := esbmc-inheritance
stream := esbmc-stream
string := esbmc-string
trycatch := esbmc-try_catch
vector := esbmc-vector

# Clean
ifneq ($(filter clean,$(MAKECMDGOALS)),)
        TARGET := clean
endif
clean: all

# Make them all
all: $(algorithm) $(cpp) $(inheritance) $(stream) $(string) $(trycatch) $(vector)

$(algorithm) $(cpp) $(inheritance) $(stream) $(string) $(trycatch) $(vector):
	$(MAKE) --directory=$@ $(TARGET)

# Make each directory
algorithm: $(algorithm)
cpp: $(cpp)
inheritance: $(inheritance)
stream: $(stream)
string: $(string)
trycatch: $(trycatch)
vector: $(vector)

# Allow parallel make
.PHONY: all $(algorithm) $(cpp) $(inheritance) $(stream) $(string) $(trycatch) $(vector)
