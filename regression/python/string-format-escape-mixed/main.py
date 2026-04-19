# Test: String format with escaped braces and args
text = "{{}} {}"
result = text.format("a")
assert result == "{} a"
assert result != "{ } a"
