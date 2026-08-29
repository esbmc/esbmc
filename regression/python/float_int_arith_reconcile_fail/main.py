# Exercises c_implicit_typecast_arithmetic through python_math's floor-div and
# modulo reconciliation with one float and one integer operand. The IREP2
# get_c_type overload used to classify floatbv as OTHER, which outranks every
# arithmetic kind and made the helper convert neither operand.
x = 7.0
assert x // 2 == 3.0
assert x % 2 == 1.0

y = 7
assert y // 2.0 == 3.0
assert y % 2.0 == 1.0

z = -7.0
assert z // 2 == -3.0
