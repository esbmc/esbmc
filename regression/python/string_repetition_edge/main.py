# Test: String repetição com número negativo
s = "abc"
result = s * 0
assert result == ""
assert len(result) == 0
result2 = s * 1
assert result2 == "abc"
result3 = s * -1  # resultado é string vazia
assert result3 == ""
