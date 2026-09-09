# functionVarargs is a set, not a signature table, so it did not cross the
# module boundary: the importing module lost the *args arity exemption and
# rejected a valid variadic call (#7543).
from vmod import take, blend


def main() -> None:
    assert take() == 7
    assert take(1, 2) == 7
    assert blend(5) == 5
    assert blend(5, 6, 7) == 5


main()
