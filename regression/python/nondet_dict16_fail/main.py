# esbmc/esbmc#7575: nondet_dict stored one pre-evaluated key on every iteration,
# so the model could never build a dict with more than one entry. An annotated
# assignment reached that model body, and this bound was proved instead of
# falsified.
d: dict[int, int] = nondet_dict(3)
assert len(d) <= 1
