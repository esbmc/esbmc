import string

split = lambda: string.split()  # noqa: E731
string = "a b"
assert len(split()) == 2
