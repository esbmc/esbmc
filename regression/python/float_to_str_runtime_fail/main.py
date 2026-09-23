# A runtime float whose repr needs more than 6 fractional digits or scientific
# notation is not rendered: the old model proved these inequalities.
def f(third: float, big: float, neg_zero: float) -> None:
    assert (
        str(third) != "0.3333333333333333"
        and str(big) != "1e+16"
        and str(neg_zero) != "-0.0"
    )


f(1 / 3, 1e16, -0.0)
