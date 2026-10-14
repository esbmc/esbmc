# Monomorphism limitation, documented rather than fixed. The frontend emits one
# GOTO function per Python function, so heapq's shared _siftup helper cannot
# serve both a tuple-keyed heap and an int heap in the same program. Here the
# two call sites disagree, the element type is left on heapq's `list[int]`
# annotation, and the tuple heap is modelled wrongly: ESBMC reports SUCCESSFUL
# although CPython raises AssertionError (heapify puts (1, 2) at the root).
#
# With only the tuple heap present this verifies correctly -- see
# github_5931_tuple_fail. When the frontend can specialise a function per call
# site, this should report FAILED and be promoted to CORE.
import heapq

pairs: list = [(1, 5), (1, 2)]
heapq.heapify(pairs)

numbers: list = [3, 1, 2]
heapq.heapify(numbers)
assert heapq.heappop(numbers) == 1

root = pairs[0]
assert root[1] == 5
