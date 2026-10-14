# A string is not a type object, although both are modelled as char arrays.
x = "int"
assert isinstance(x, type)
