# Test: nondet_string com operador in e assume - SUCESSO
from esbmc import nondet_string, assume

s = nondet_string(11)
assume(s == "hello world")
assert "world" in s
assert "hello" in s
assert "xyz" not in s
