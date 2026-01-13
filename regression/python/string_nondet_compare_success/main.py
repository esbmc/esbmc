# Test: nondet_string com assume de diferença - SUCESSO
from esbmc import nondet_string, assume

s = nondet_string(4)
assume(s != "test")
assert s != "test"
# s pode ser qualquer coisa menos "test"
