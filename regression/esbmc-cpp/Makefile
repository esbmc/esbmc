# Directories
algorithm := algorithm
cpp := cpp
inheritance := inheritance
stream := stream
string := string
trycatch := try_catch
vector := vector
deque := deque
list := list
stack := stack
queue := queue
priority_queue := priority_queue
map := map
set := set
multimap := multimap
multiset := multiset

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
all: $(algorithm) $(cpp) $(inheritance) $(stream)  $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(string)

$(algorithm) $(cpp) $(inheritance) $(stream) $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set)  $(multimap) $(multiset) $(string):
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

.PHONY: all $(algorithm) $(cpp) $(inheritance) $(stream) $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(string)
