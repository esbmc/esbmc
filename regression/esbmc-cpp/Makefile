# Directories for esbmc-cpp
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
unordered_map := unordered_map
functional := functional
<<<<<<< HEAD
bitset := bitset

=======
>>>>>>> e7c955101 (Update stats-300s.txt)

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
all: $(algorithm) $(cpp) $(inheritance) $(stream)  $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(string) $(unordered_set) $(unordered_map) $(functional)

$(algorithm) $(cpp) $(inheritance) $(stream) $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set)  $(multimap) $(multiset) $(string) $(unordered_set) $(unordered_map) $(functional):
	$(MAKE) --directory=$@ $(TARGET)

#Make only STL directories
stl: $(algorithm) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(unordered_set) $(unordered_map) $(functional)
<<<<<<< HEAD

=======
>>>>>>> e7c955101 (Update stats-300s.txt)

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
unordered_map: $(unordered_map)
functional : $(functional)
<<<<<<< HEAD
bitset : $(bitset)
=======
>>>>>>> e7c955101 (Update stats-300s.txt)

# Allow parallel make

.PHONY: all $(algorithm) $(cpp) $(inheritance) $(stream) $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(string) $(unordered_set) $(unordered_map) $(functional)
