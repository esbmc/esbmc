# bytearray is not modeled. Before the fix this fell through to an empty_typet()
# that later crashed symex with an uncaught type2t::symbolic_type_excp (SIGABRT)
# on item assignment. It must now produce a clean "not supported" diagnostic.
b = bytearray([1, 2, 3])
b[0] = 9
assert b[0] == 9
