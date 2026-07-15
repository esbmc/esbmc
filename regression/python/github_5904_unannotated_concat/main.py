# Regression for issue #5904: the TypeError guard for `str + <non-str>` must NOT
# misfire on an unannotated operand whose real type is str. `x` here is
# unannotated (any_type), so its category is unknown; concatenation must still
# succeed rather than raise a spurious TypeError.
def f(x) -> str:
    return x + "!"


def main():
    r = f("hey")
    assert len(r) == 4


main()
