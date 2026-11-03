# Issue #9444: where no runtime check can be planted (an and/or operand), the
# unsupported round(x, ndigits) stops conversion instead of crashing.
x: int = nondet_int()
b: bool = nondet_bool()
c = b and round(x, -1) == 0
