def f(d: dict[str, int]) -> None:
    for k, v in d.items():
        assert v >= 0

# assume symbolic d
x = nondet_dict()
f(x)
