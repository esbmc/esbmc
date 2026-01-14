# Test: nondet_string com assume duplo - SUCESSO
from esbmc import nondet_string, assume

s1 = nondet_string(3)
s2 = nondet_string(3)
assume(s1 == "abc")
assume(s2 == "abc")
assert s1 == s2
assert s1 == "abc"
