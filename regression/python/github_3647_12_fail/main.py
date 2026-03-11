def f(d: dict[str, dict[str, int]]) -> None:
    for k1, inner in d.items():
        for k2, v in inner.items():
            assert v < 0

f({"a": {"b": 1}})
