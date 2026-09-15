# esbmc/esbmc#7575: nondet_list appended one pre-evaluated value to every index,
# so two indices could never differ. An annotated assignment reached that model
# body, and this assertion was proved instead of falsified.
x: list[int] = nondet_list(3)
if len(x) == 2:
    assert x[0] == x[1]
