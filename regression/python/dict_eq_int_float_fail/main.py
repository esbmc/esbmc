# Negative variant of dict_eq_int_float: a genuinely different float value
# (1.5 vs 1) must still compare unequal -- the numeric int/float fallback
# must not paper over real differences.
def main() -> None:
    d1 = {"a": 1.5, "b": 2.0}
    d2 = {"a": 1, "b": 2}
    assert d1 == d2


main()
