def f(d: dict[str, int]) -> None:
    for k, v in d.items():
        assert isinstance(v, str)

d = {"a": 1}
f(d)
