# Negative companion: when join() falls back to a nondet `char *`, an
# assertion that pins a specific result must reach the solver and
# fail rather than silently succeeding.


def join_sorted(parts):
    return ' '.join(sorted(parts))


if __name__ == "__main__":
    # The nondet fallback can be any string; the strcmp against a
    # specific literal is falsifiable.
    result = join_sorted(["a", "b"])
    assert result == "a b"
