# Test: String com nondet e operações - deve FALHAR sem assume
from esbmc import nondet_string

s = nondet_string(5)
# Sem assume, não podemos garantir nada sobre a string
assert len(s) == 5  # Isso deve passar (tamanho é conhecido)
assert s[0] == "a"  # Isso deve FALHAR (valor não determinístico)
