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
queue := esbmc-queue
priority_queue := esbmc-priority_queue
map := esbmc-map
set := esbmc-set
multimap := esbmc-multimap
multiset := esbmc-multiset

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
all: $(algorithm) $(cpp) $(inheritance) $(stream) $(string) $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset)

$(algorithm) $(cpp) $(inheritance) $(stream) $(string) $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set)  $(multimap) $(multiset):
	$(MAKE) --directory=$@ $(TARGET)

#Make only STL directories
stl: $(algorithm) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset)

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
queue: $(queue) 
priority_queue: $(priority_queue)
map: $(map)
set: $(set)
multimap: $(multimap)
multiset: $(multiset)

# Allow parallel make

.PHONY: all $(algorithm) $(cpp) $(inheritance) $(stream) $(string) $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset)
