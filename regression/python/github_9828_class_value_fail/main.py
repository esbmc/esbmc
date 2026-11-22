import to_import

x = to_import.A(3)
a = to_import.A(5)
assert isinstance(a, to_import.A)
b = a.b
assert isinstance(b, int)
assert b + x.b == 9
