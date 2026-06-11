# tuple() over a non-list, non-tuple iterable (here a string) is not
# modelled. ESBMC must reject it with an explicit error rather than relabel
# the operand with an empty type (which previously made every comparison
# over the result silently lower to a nondet bool). This program is valid
# CPython (s becomes ('a', 'b')).
s = tuple("ab")
