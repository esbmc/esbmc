# check_type must not raise when the value's own type is not tracked: `b` here
# comes from an indirect call through a function-valued parameter, and
# isinstance() folds to false for "cannot tell" as well as for a proven
# mismatch. Raising on the first case is a false alarm on a correct program.
import _sv_verifier


def returns_int():
    return 1


def call(f):
    return f()


def main():
    b = call(returns_int)
    _sv_verifier.check_type(b, int)


main()
