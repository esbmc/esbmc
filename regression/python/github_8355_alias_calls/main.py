# A conditional expression whose arms call a function through an alias
# (#8355).
def f(x: int) -> int:
    return x


g = f
n = nondet_int()
c = g(n) if nondet_bool() else g(1)
assert c == n or c == 1
