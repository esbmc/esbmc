# Negative companion: pin the runtime model's actual semantics. A
# wrong expected result must reach the solver and fail rather than
# spuriously succeed (which would happen if the handler quietly fell
# back to the nondet string and the comparison silently held under
# some unconstrained value).


def flip(s: str) -> str:
    return s.swapcase()


if __name__ == "__main__":
    assert flip("Hello") == "Hello"  # actual is "hELLO"; must fail
