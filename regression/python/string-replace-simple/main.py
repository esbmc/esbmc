# Test: String replace simples
text = "hello world"
result = text.replace("world", "Python")
assert result == "hello Python"
result2 = text.replace("l", "L")
assert result2 == "heLLo worLd"
result3 = text.replace("xyz", "abc")
assert result3 == "hello world"  # não mudou
