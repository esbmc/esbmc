# The early type-object check read only a bare name, so a tuple of types
# folded to false and this assertion was proved.
x = int
assert not isinstance(x, (type, str))
