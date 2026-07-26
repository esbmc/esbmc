# Known frontend defect (issue #5931 follow-up): tuple elements in a heap are
# typed as int by the Python frontend, so the heap layout is modelled wrongly
# and ESBMC currently reports VERIFICATION SUCCESSFUL for this program even
# though CPython's heapify() puts (1, 2) at the root and the assertion fails.
#
# This reproduces identically against the pre-#5931 heapq model, so it is not a
# regression introduced by the faithful CPython port. When the frontend learns
# to type tuple elements, ESBMC will report VERIFICATION FAILED here, this test
# will start matching its expected output, and it should be promoted to CORE.
import heapq

h: list = [(1, 5), (1, 2)]
heapq.heapify(h)
x = h[0]
assert x[1] == 5
