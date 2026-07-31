
s = nondet_str()
result = s + "def"
assert result == "abcdef"  # FALHA - s é não determinístico
