# str elements are not modelled by the Python frontend. ESBMC currently reports
# VERIFICATION FAILED here (a loud false alarm) even though the assertion holds
# in CPython. That is sound -- unlike the tuple case, it never yields a false
# proof -- but it is a real limitation. When str heaps are supported, ESBMC will
# report SUCCESSFUL, this test will match, and it should be promoted to CORE.
import heapq

h: list[str] = ["b", "a"]
heapq.heapify(h)
assert h[0] == "a"
