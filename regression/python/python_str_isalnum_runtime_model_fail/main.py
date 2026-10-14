# Negative companion: pin the runtime model's actual semantics. A
# wrong expected predicate must reach the solver and fail rather than
# spuriously succeed (which would happen if the handler quietly
# returned nondet).


def is_alnum(s: str) -> bool:
    return s.isalnum()


if __name__ == "__main__":
    assert is_alnum("hello!") == True  # actually False; must fail
