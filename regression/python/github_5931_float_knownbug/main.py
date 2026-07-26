# Float heaps are not fully modelled. The frontend types the model's heap
# elements as int (see models/heapq.py), so heapify() and plain index reads on a
# float heap are exact, but any value-returning operation (heappop, heapreplace,
# heappushpop) yields an unconstrained result and heappush() of a float item
# corrupts the heap. ESBMC therefore reports VERIFICATION FAILED here even
# though the assertions hold in CPython.
#
# This is a loud false alarm, never a false proof: no float program was found
# for which ESBMC reports SUCCESSFUL while CPython's assertion fails. When float
# elements are modelled, ESBMC will report SUCCESSFUL, this test will match its
# expected output, and it should be promoted to CORE.
import heapq

h: list[float] = [3.5, 1.5, 2.5]
heapq.heapify(h)
assert h[0] == 1.5
assert heapq.heappop(h) == 1.5
