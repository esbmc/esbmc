# isinstance negative cases: non-complex types are not complex.

x_int = 42
assert not isinstance(x_int, complex)

x_float = 3.14
assert not isinstance(x_float, complex)

x_bool = True
assert not isinstance(x_bool, complex)
