# Negative variant of github_4715_irep2_bodies_py_om_02: a wrong expected value
# must still produce a counterexample (VERIFICATION FAILED) under --irep2-bodies,
# proving the dict subscript read is genuinely constrained.
d: dict[int, float] = {1: 1.0}
d[2] = 3.5
assert d[2] == 4.0
