def f(d: dict[str, int]) -> None:
    for k, v in d.items():
        assert False  # should never execute

f({})
