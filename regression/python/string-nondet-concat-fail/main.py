
s = nondet_string(3)
result = s + "def"
assert result == "abcdef"  # FALHA - s é não determinístico
