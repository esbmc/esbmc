# Test: nondet_string métodos com assume - SUCESSO
from esbmc import nondet_string, assume

s = nondet_string(5)
assume(s == "hello")
upper = s.upper()
assert upper == "HELLO"
assert len(upper) == 5
