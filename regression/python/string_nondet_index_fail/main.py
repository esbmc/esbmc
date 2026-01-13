# Test: nondet_string indexação sem assume - FALHA
from esbmc import nondet_string

s = nondet_string(5)
c = s[0]
assert c == "h"  # FALHA - valor não determinístico
