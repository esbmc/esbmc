
s = nondet_string(3)
c = s[0]  # caractere não determinístico
assert c == "a"  # FALHA - não sabemos o valor
