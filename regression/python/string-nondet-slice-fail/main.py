s = nondet_string(6)
sub = s[0:3]
assert sub == "hel"  # FALHA - não sabemos o conteúdo
