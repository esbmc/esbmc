# Test: nondet_string slicing sem assume - FALHA
from esbmc import nondet_string

s = nondet_string(6)
sub = s[0:3]
assert sub == "hel"  # FALHA - não sabemos o conteúdo
