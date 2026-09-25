# clang_c_adjust rewrites float +,-,*,/ to ieee_add/sub/mul/div and attaches the
# rounding mode; python_adjust had no counterpart, so under the hop-off a float
# `+` stayed a plain add2t and tripped simplify_arith_2ops'
# assert(!is_floatbv_type(type)) -- "this should be handled by ieee_*".
# Exercises all four operators plus the mixed int/float promotion that reaches
# them (scope-coupled-arith-assign-conversion.md §17).
a: float = 1.5
b: float = 0.5
assert a + b == 2.0
assert a - b == 1.0
assert a * b == 0.75
assert a / b == 3.0
assert sum((1, 2.5, 3)) == 6.5
