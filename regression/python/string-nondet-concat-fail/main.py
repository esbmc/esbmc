# Test: nondet_string concatenação sem assume - FALHA
from esbmc import nondet_string

s = nondet_string(3)
result = s + "def"
assert result == "abcdef"  # FALHA - s é não determinístico
