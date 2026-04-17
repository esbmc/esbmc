def f(d: dict[str, int]) -> None:
    for pair in d.items():
        # In real Python, pair is a tuple
        assert pair is not None

d = {"a": 1}
f(d)
