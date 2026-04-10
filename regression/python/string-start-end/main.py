# Test: String startswith e endswith
text = "hello world"
assert text.startswith("hello")
assert text.endswith("world")
assert not text.startswith("world")
assert not text.endswith("hello")
assert text.startswith("h")
