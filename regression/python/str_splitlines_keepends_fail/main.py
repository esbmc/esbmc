# With keepends the newline is retained, so the parts are not bare "a"/"b".
assert "a\nb".splitlines(True) == ["a", "b"]
