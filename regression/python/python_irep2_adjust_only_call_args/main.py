# Exercises the --python-irep2-adjust-only conversion of an expression-form
# call's arguments to the declared parameter types. Only the expression form
# converts them: the legacy statement-form arm adjusts index expressions and
# nothing else, so a call feeding a larger expression is the shape that must
# still reach its callee under the declared signature.
def scale(x: float) -> float:
    return x * 2.0


def main() -> None:
    assert scale(3) + scale(4) == 14.0


main()
