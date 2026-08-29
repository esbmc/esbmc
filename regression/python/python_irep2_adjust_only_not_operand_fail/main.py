# Negative counterpart: the same non-Boolean `not` operand with a false
# assertion. The cast must let the negation reach the solver and report the
# violation, not abort the encoding.
x = None

assert not (x or True)
