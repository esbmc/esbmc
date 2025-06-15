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
unordered_set := unordered_set

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
all: $(algorithm) $(cpp) $(inheritance) $(stream)  $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(string) $(unordered_set)

$(algorithm) $(cpp) $(inheritance) $(stream) $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set)  $(multimap) $(multiset) $(string) $(unordered_set):
	$(MAKE) --directory=$@ $(TARGET)

#Make only STL directories
stl: $(algorithm) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(unordered_set)

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
unordered_set: $(unordered_set)

# Allow parallel make

.PHONY: all $(algorithm) $(cpp) $(inheritance) $(stream) $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(string) $(unordered_set)
