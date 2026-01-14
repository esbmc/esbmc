# Test: nondet_string com operador in sem assume - FALHA
from esbmc import nondet_string

s = nondet_string(10)
assert "test" in s  # FALHA - não sabemos o conteúdo
