def f(d: dict[str, int]) -> None:
    for k, v in d.items():
        d["x"] = 3
        assert True

f({"a": 1})
