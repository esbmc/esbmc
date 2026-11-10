class C:
    @staticmethod
    def interfaces() -> dict[str, dict[str, str]]:
        return {"a": {}, "b": {"flags": "up"}}


def f() -> None:
    seen = 0
    for name, iface in C.interfaces().items():
        x = iface.get('flags')
        if name == "b":
            assert x is not None
        else:
            assert x is None
        seen += 1
    assert seen == 2


f()
