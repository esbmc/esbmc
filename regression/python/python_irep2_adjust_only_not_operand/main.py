# Exercises the --python-irep2-adjust-only bool cast on a `not` operand.
# `x and True` over a None-valued `x` lowers to a pointer-typed short-circuit
# select, so `not (x and True)` negates a non-Boolean value. clang_c_adjust casts
# the operand (adjust_expr_unary_boolean); without the cast the negation reaches
# the SMT layer over a bitvector sort and bitwuzla aborts in mk_not.
x = None

assert not (x and True)
assert x or True
assert (x or 2) == 2
assert not (x and 1)
