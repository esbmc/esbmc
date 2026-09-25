# A string is not a type object, although both are modelled as char arrays.
assert not isinstance("int", type)
assert not isinstance("hello", type)
