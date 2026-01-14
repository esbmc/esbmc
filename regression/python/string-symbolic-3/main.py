# Test: Concatenação com string simbólica - deve PASSAR
from esbmc import nondet_string, assume

s1 = nondet_string(3)
assume(s1 == "abc")
s2 = "def"
result = s1 + s2
assert result == "abcdef"
assert len(result) == 6
