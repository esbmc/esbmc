# Falsification harness for heapq.heapify (src/python-frontend/models/heapq.py).
#
# heapify builds a MIN-heap, so the root is the smallest element, never the
# largest. Asserting a max-heap root must be falsifiable.
#
# REQUIRES:
#   R1: three fully non-deterministic integers.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: h[0] >= h[1] after heapify.  False whenever the left child is strictly
#       larger than the root (a min-heap), e.g. elements 1, 2, 3.
import heapq

a: int = nondet_int()
b: int = nondet_int()
c: int = nondet_int()
h: list[int] = [a, b, c]

heapq.heapify(h)

assert h[0] >= h[1]     # F1 — falsifiable (min-heap root is the smallest)
