# Test: nondet_string tamanho - SUCESSO
from esbmc import nondet_string

s = nondet_string(10)
# Tamanho é conhecido
assert len(s) == 10
# Mas valores não são
