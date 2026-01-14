# Test: String index com substring não encontrada - deve FALHAR
text = "hello"
pos = text.index("xyz")  # ValueError - não encontrado
assert pos == -1
