# Test: Comparação de duas strings simbólicas - deve FALHAR
from esbmc import nondet_string

s1 = nondet_string(3)
s2 = nondet_string(3)
# Sem assumes, não podemos garantir que sejam iguais
assert s1 == s2  # Deve FALHAR
