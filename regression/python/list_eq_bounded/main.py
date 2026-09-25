# Pins what the list-equality operational model computes, at a bound. Its
# worklist loop is not structurally bounded (see
# docs/roadmap/scope-list-eq-unbounded-unwinding.md), so this runs under
# --unwind; a fix that bounds the loop must not change these verdicts.
m: list = ["y"]
n: list = ["y"]
o: list = ["z"]

assert m == n
assert not (m == o)
assert m != o
