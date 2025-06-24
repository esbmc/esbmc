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
<<<<<<< HEAD
<<<<<<< HEAD
unordered_map := unordered_map
functional := functional
<<<<<<< HEAD
<<<<<<< HEAD
bitset := bitset

=======
>>>>>>> e7c955101 (Update stats-300s.txt)
=======
<<<<<<< HEAD
=======
>>>>>>> 0cd441df3 ([c++] add basic unordered_set support (#2529))
=======
unordered_map := unordered_map
>>>>>>> c76c4bd02 ([c++] added basic std::unordered_map support (#2530))
=======
>>>>>>> fa0da2b3d ([c++] add deterministic hash specializations for std::hash (#2538))
=======
bitset := bitset
>>>>>>> dd20ccf75 ([c++] added basic bitset operational model (#2543))
>>>>>>> a710fa95b (Improved memcpy function)

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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
all: $(algorithm) $(cpp) $(inheritance) $(stream)  $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(string) $(unordered_set) $(unordered_map) $(functional)
=======
all: $(algorithm) $(cpp) $(inheritance) $(stream)  $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(string) $(unordered_set) $(unordered_map) $(functional) $(bitset)
>>>>>>> dd20ccf75 ([c++] added basic bitset operational model (#2543))

$(algorithm) $(cpp) $(inheritance) $(stream) $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set)  $(multimap) $(multiset) $(string) $(unordered_set) $(unordered_map) $(functional) $(bitset):
	$(MAKE) --directory=$@ $(TARGET)

#Make only STL directories
<<<<<<< HEAD
stl: $(algorithm) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(unordered_set) $(unordered_map) $(functional)
=======
all: $(algorithm) $(cpp) $(inheritance) $(stream)  $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(string) $(unordered_set)
=======
all: $(algorithm) $(cpp) $(inheritance) $(stream)  $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(string) $(unordered_set) $(unordered_map)
>>>>>>> c76c4bd02 ([c++] added basic std::unordered_map support (#2530))
=======
all: $(algorithm) $(cpp) $(inheritance) $(stream)  $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(string) $(unordered_set) $(unordered_map) $(functional)
>>>>>>> fa0da2b3d ([c++] add deterministic hash specializations for std::hash (#2538))

$(algorithm) $(cpp) $(inheritance) $(stream) $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set)  $(multimap) $(multiset) $(string) $(unordered_set) $(unordered_map) $(functional):
	$(MAKE) --directory=$@ $(TARGET)

#Make only STL directories
<<<<<<< HEAD
<<<<<<< HEAD
stl: $(algorithm) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(unordered_set)
>>>>>>> 0cd441df3 ([c++] add basic unordered_set support (#2529))
=======
stl: $(algorithm) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(unordered_set) $(unordered_map)
>>>>>>> c76c4bd02 ([c++] added basic std::unordered_map support (#2530))
=======
stl: $(algorithm) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(unordered_set) $(unordered_map) $(functional)
<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> e7c955101 (Update stats-300s.txt)
=======
>>>>>>> fa0da2b3d ([c++] add deterministic hash specializations for std::hash (#2538))
=======
stl: $(algorithm) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(unordered_set) $(unordered_map) $(functional) $(bitset)
>>>>>>> dd20ccf75 ([c++] added basic bitset operational model (#2543))
>>>>>>> a710fa95b (Improved memcpy function)

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
<<<<<<< HEAD
<<<<<<< HEAD
unordered_map: $(unordered_map)
functional : $(functional)
bitset : $(bitset)

# Allow parallel make

.PHONY: all $(algorithm) $(cpp) $(inheritance) $(stream) $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(string) $(unordered_set) $(unordered_map) $(functional)
=======

# Allow parallel make

.PHONY: all $(algorithm) $(cpp) $(inheritance) $(stream) $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(string) $(unordered_set)
>>>>>>> 0cd441df3 ([c++] add basic unordered_set support (#2529))
=======
unordered_map: $(unordered_map)
functional : $(functional)
<<<<<<< HEAD
bitset : $(bitset)
=======
>>>>>>> e7c955101 (Update stats-300s.txt)

# Allow parallel make

<<<<<<< HEAD
.PHONY: all $(algorithm) $(cpp) $(inheritance) $(stream) $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(string) $(unordered_set) $(unordered_map)
>>>>>>> c76c4bd02 ([c++] added basic std::unordered_map support (#2530))
=======
.PHONY: all $(algorithm) $(cpp) $(inheritance) $(stream) $(trycatch) $(vector) $(deque) $(list) $(stack) $(queue) $(priority_queue) $(map) $(set) $(multimap) $(multiset) $(string) $(unordered_set) $(unordered_map) $(functional)
>>>>>>> fa0da2b3d ([c++] add deterministic hash specializations for std::hash (#2538))
