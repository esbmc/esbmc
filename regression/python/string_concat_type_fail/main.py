# Test: Concatenação com tipo errado - deve FALHAR
s = "hello"
num = 123
result = s + num  # TypeError
assert result == "hello123"
