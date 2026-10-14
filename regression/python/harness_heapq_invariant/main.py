# Verification harness for heapq.heapify (src/python-frontend/models/heapq.py).
#
# heapify(h) rearranges the list in place into a binary min-heap, where every
# parent is <= its children: for index k, h[k] <= h[2k+1] and h[k] <= h[2k+2]
# (when those child indices are in range). The model reproduces CPython's sift
# routines element-for-element. Only int elements are fully sound (the model
# header documents float/str as "loud" and tuple as unsound), so the harness
# uses int elements.
#
# REQUIRES:
#   R1: three fully non-deterministic integers seed the heap, covering every
#       ordering of the elements.
#
# ENSURES (after heapify of length-3 h, with root 0 and children 1, 2):
#   E1: h[0] <= h[1]                             [root dominates left child]
#   E2: h[0] <= h[2]                             [root dominates right child]
import heapq

a: int = nondet_int()
b: int = nondet_int()
c: int = nondet_int()
h: list[int] = [a, b, c]

heapq.heapify(h)

assert h[0] <= h[1]  # E1
assert h[0] <= h[2]  # E2
