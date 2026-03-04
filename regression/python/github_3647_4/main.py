def f(d: dict[str, int]) -> None:
    for k, v in d.items():
        assert isinstance(v, int)

d = {"a": 1}
f(d)
