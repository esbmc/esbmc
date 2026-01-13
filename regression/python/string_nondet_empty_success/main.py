# Test: nondet_string vazio - SUCESSO
from esbmc import nondet_string

s = nondet_string(0)
assert len(s) == 0
assert s == ""
result = s + "test"
assert result == "test"
