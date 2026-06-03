# A walrus inside an `and`/`or` operand may be short-circuited away at runtime,
# so binding it unconditionally would be unsound. ESBMC refuses with a clean
# diagnostic. (CPython: a is truthy, so `(b := 99)` never runs and b stays 0.)
a = 7
b = 0
z = a or (b := 99)
assert b == 0
