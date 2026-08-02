# Negative variant of github_4715_irep2_bodies_py_om_01: a wrong expected value
# must still produce a counterexample (VERIFICATION FAILED) under --irep2-bodies,
# proving the element read is genuinely constrained and not masked by an
# alignment crash.
l = [10, 20, 30, 40, 50]
l.append(60)
assert l[2] == 31
