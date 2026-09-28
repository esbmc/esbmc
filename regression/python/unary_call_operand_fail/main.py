# A unary operator over a call kept the call as a statement nested in the
# expression, which reached the solver unevaluated and crashed it.
def f(x: float) -> float:
    return x * 2


def h(a: int) -> int:
    return a


assert -f(1.25) == 2.5 or ~h(5) == 6
