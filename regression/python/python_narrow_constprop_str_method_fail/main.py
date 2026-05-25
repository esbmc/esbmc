# Negative companion: with more than one assignment to `s`, the narrow
# propagation must bail (no flow-sensitive reasoning), so count() falls
# back to nondet and an exact-value assertion can be falsified.


def f(x: bool) -> int:
    if x:
        s = "ab"
    else:
        s = "cd"
    return s.count('a')


if __name__ == "__main__":
    # If we wrongly folded one branch the result would be deterministic
    # (0 or 1). With the bail in place, count() is nondet -- the
    # assertion below is falsifiable.
    assert f(True) == 99
