# Runtime operational model for str.swapcase.
#
# Before this change, handle_string_swapcase required a compile-time
# constant receiver; non-constant receivers fell back to
# build_nondet_string_fallback (sound but unconstrained -- subsequent
# comparisons against the result could never hold exactly).
#
# __python_str_swapcase(const char *) -- added to
# src/c2goto/library/python/string.c -- is a bounded ASCII case-swap
# (255 char cap, longer strings trip an explicit assertion). The
# handler now dispatches to it for non-constant receivers; symex's
# existing constant propagation folds the call on concrete arguments.
#
# Pins exact-value assertions through a parameterised wrapper -- only
# pass if the model actually produces the swapped string.


def flip(s: str) -> str:
    return s.swapcase()


if __name__ == "__main__":
    assert flip("Hello") == "hELLO"
    assert flip("ABC") == "abc"
    assert flip("abc") == "ABC"
    assert flip("aB1!cD") == "Ab1!Cd"
    assert flip("") == ""
