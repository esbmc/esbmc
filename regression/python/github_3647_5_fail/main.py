def f(d: dict[str, int]) -> None:
    for k, v in d.items():
        assert v > 100

d = {"a": 1}
f(d)
