# Test: nondet_string básico sem assume - FALHA
from esbmc import nondet_string

s = nondet_string(5)
# Sem assume, não podemos garantir o valor
assert s == "hello"  # FALHA
