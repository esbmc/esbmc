# Test: nondet_string métodos sem assume - FALHA
from esbmc import nondet_string

s = nondet_string(5)
upper = s.upper()
assert upper == "HELLO"  # FALHA
