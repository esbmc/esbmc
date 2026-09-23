# A unary operator over a call kept the call as a statement nested in the
# expression, which reached the solver unevaluated and crashed it; `~` also
# took the enclosing assert's bool type.
cnt = 0


def f(x: float) -> float:
    global cnt
    cnt += 1
    return x * 2


def g() -> bool:
    return False


def h(a: int) -> int:
    return a


assert -f(1.25) == -2.5
assert cnt == 1
assert not g()
assert ~h(5) == -6
assert -f(-f(1.0)) == 4.0
