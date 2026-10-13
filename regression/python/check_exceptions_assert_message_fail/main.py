# A Python assert whose message names an exception type still raises
# AssertionError, so it is not mistaken for a model check of KeyError.
def main() -> None:
    i = nondet_int()
    assert i > 0, "KeyError: i must be positive"


main()
