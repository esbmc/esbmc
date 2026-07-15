# Regression for the false proof in issue #5931: the old operational model
# left heapify() as a no-op, so ESBMC reported VERIFICATION SUCCESSFUL for
# this program even though CPython raises AssertionError.
import heapq

items: list[int] = [3, 1, 2]
heapq.heapify(items)
assert items[0] == 3
