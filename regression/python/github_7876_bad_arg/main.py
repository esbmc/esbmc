# esbmc/esbmc#7876: accepting any member of an opaque union must not accept
# everything. A float is in neither arm of `int | list[int]`, so --strict-types
# still raises TypeError (as it did before the fix). The nondet argument keeps
# consteval from folding the assertion away before the call is checked.
def foo(y: int | list[int]) -> int:
    return 1


x = nondet_float()

assert foo(x) == 1
