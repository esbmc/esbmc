# Test: String vazia simbólica - deve PASSAR
from esbmc import nondet_string, assume

s = nondet_string(0)  # String vazia
assert len(s) == 0
assert s == ""
empty = s + "test"
assert empty == "test"
