# Negative companion: pin the runtime model's actual semantics. A wrong
# expected count must reach the solver and fail rather than spuriously
# succeed (which is what would happen if the handler quietly returned
# nondet again).


def count_in(s: str, sub: str) -> int:
    return s.count(sub)


if __name__ == "__main__":
    assert count_in("hello", "l") == 99  # actual is 2; must fail
