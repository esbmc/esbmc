# Test: String não-determinística - deve FALHAR
from esbmc import nondet_string

s = nondet_string(5)  # string simbólica de tamanho 5
# Não podemos garantir que seja "hello" sem constraint
assert s == "hello"  # Deve FALHAR - string pode ser qualquer coisa
