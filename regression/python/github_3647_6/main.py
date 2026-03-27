def make() -> dict[str, int]:
    return {"a": 1}

def f() -> None:
    for k, v in make().items():
        assert v == 1

f()
