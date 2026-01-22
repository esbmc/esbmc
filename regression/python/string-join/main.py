# Test: String join
parts = ["hello", "world"]
result = " ".join(parts)
assert result == "hello world"
parts2 = ["a", "b", "c"]
result2 = "-".join(parts2)
assert result2 == "a-b-c"
