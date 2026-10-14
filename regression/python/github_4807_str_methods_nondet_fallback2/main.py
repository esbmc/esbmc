# Regression for #4807: a second batch of str.*() methods that previously
# aborted GOTO conversion with "X() requires constant string" on a non-
# constant receiver. The handlers now fall back to a sound symbolic value
# via string_handler::build_nondet_string_fallback (string-returning) or
# side_effect_expr_nondett(bool_type()) (predicate-returning), so the
# function bodies convert and verification proceeds.
#
# Methods covered here: removeprefix, removesuffix, isnumeric, isidentifier,
# center, ljust, rjust, zfill, expandtabs.
#
# Each wrapper takes a non-constant parameter and exercises one method. The
# call sites are not asserted on — the test only proves that conversion
# completes without aborting, which is the regression this PR closes.


def f_removeprefix(s: str, p: str) -> str:
    return s.removeprefix(p)


def f_removesuffix(s: str, x: str) -> str:
    return s.removesuffix(x)


def f_isnumeric(s: str) -> bool:
    return s.isnumeric()


def f_isidentifier(s: str) -> bool:
    return s.isidentifier()


def f_center(s: str, w: int) -> str:
    return s.center(w)


def f_ljust(s: str, w: int) -> str:
    return s.ljust(w)


def f_rjust(s: str, w: int) -> str:
    return s.rjust(w)


def f_zfill(s: str, w: int) -> str:
    return s.zfill(w)


def f_expandtabs(s: str) -> str:
    return s.expandtabs(4)


if __name__ == "__main__":
    # No call sites: the function bodies above are emitted into GOTO and
    # previously aborted conversion. Successful conversion + zero VCCs ==
    # verification succeeds.
    pass
