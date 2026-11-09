import to_import

x = to_import.A()
a = to_import.A(5)
assert isinstance(a, to_import.A)
b = a.b
assert isinstance(b, int)
