# Test: Caractere individual - operações básicas - SUCESSO
c = "a"
assert c == "a"
assert len(c) == 1
assert c != "b"
assert c < "b"
assert c > "Z"  # lowercase > uppercase em ASCII
# Concatenação
c2 = c + "b"
assert c2 == "ab"
# Repetição
c3 = c * 3
assert c3 == "aaa"
# Comparação
assert "a" < "z"
assert "A" < "a"  # uppercase < lowercase
