# Regression for the second false proof in issue #5931: the old model's
# heappop() scanned for the minimum, but CPython's heappop() returns the
# root. On a list that is not a valid heap the two disagree, and ESBMC
# reported VERIFICATION SUCCESSFUL where CPython raises AssertionError.
import heapq

h: list[int] = [3, 1, 2]
assert heapq.heappop(h) == 1
