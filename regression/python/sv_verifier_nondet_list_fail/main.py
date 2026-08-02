# Companion to sv_verifier_nondet_list: nondet_list(nondet_int) yields a list of
# nondeterministic ints, so asserting a concrete element value must fail.
from _sv_verifier import nondet_int, nondet_list


def main() -> None:
    xs = nondet_list(nondet_int)
    for x in xs:
        assert x == 0


main()
