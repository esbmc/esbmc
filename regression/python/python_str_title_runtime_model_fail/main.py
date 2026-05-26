# Negative companion: pin the runtime model's actual semantics. A
# wrong expected result must reach the solver and fail rather than
# spuriously succeed (which would happen if the dispatch silently fell
# back to the nondet string and the comparison silently held under
# some unconstrained value).


def t(s: str, do_it: bool) -> str:
    if do_it:
        return s.title()
    return s


if __name__ == "__main__":
    # Actual title of "hello world" is "Hello World"; the assertion is
    # contradictory and must FAILED.
    assert t("hello world", True) == "hello world"
