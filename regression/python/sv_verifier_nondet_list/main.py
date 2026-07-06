# SV-COMP's _sv_verifier.nondet_list(nondet_elem) passes the element generator
# as the first positional argument. ESBMC's built-in nondet_list model used to
# read that argument as a max_size, comparing an int against a function pointer
# and crashing during SMT encoding. The model now recognises the SV-COMP
# convention. Every element is a nondet int, so the size is bounded and the
# collection is non-None: VERIFICATION SUCCESSFUL.
from _sv_verifier import nondet_int, nondet_list


def main() -> None:
    xs = nondet_list(nondet_int)
    assert xs is not None


main()
