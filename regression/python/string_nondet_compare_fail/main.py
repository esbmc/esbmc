# Test: nondet_string comparação com literal - FALHA
from esbmc import nondet_string

s = nondet_string(4)
assert s != "test"  # FALHA - pode ser "test"
