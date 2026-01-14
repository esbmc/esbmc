# Test: String simbólica com assume - deve PASSAR
from esbmc import nondet_string, assume

s = nondet_string(5)
assume(s == "world")  # Constraint: s deve ser "world"
assert s == "world"  # Deve PASSAR
assert len(s) == 5
