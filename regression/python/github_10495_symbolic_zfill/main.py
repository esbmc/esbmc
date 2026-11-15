# Symbolic widths must not silently produce a nondeterministic string (#10495).
n: int = nondet_int()
if -3 <= n <= -1:
    assert "ab".zfill(n) == "ab"
