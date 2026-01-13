# Test: nondet_string com assume - SUCESSO
from esbmc import nondet_string, assume

s = nondet_string(5)
assume(s == "hello")
assert s == "hello"
assert len(s) == 5
