
s = nondet_string(5)
assert len(s) == 5  # Isso deve passar (tamanho é conhecido)
assert s[0] == "a"  # Isso deve FALHAR (valor não determinístico)
