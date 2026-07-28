# Negative counterpart: the same expression-form calls with a false assertion.
# The converted arguments must reach the solver and the violation be reported.
def scale(x: float) -> float:
    return x * 2.0


def main() -> None:
    assert scale(3) + scale(4) == 99.0


main()
