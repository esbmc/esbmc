# Negative counterpart: the same mixed signedbv/fixedbv product, with a false
# assertion. The reconciled operands must still reach the solver and the
# violation must be reported, so the arm cannot mask a genuine failure.


def mix(n: int, x: float) -> float:
    return n * x + n


assert mix(2, 1.5) == 4.0
