# Negative companion: pin the runtime model's actual semantics. A
# wrong expected result must reach the solver and fail rather than
# spuriously succeed (which would happen if the handler quietly fell
# back to the nondet string and the comparison silently held under
# some unconstrained value).


def capitalize(s: str) -> str:
    return s.capitalize()


if __name__ == "__main__":
    assert capitalize("hello") == "hello"  # actual is "Hello"; must fail
