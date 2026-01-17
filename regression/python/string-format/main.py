# Test: String format
name = "Python"
version = 3
result = "{} {}".format(name, version)
assert result == "Python 3"
result2 = "Hello {}".format("World")
assert result2 == "Hello World"
