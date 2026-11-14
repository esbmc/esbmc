class C:
    def interfaces(self) -> dict[str, dict[str, str]]:
        return {"a": {"flags": "up"}}


def f() -> None:
    c = C()
    for name, iface in c.interfaces().items():
        assert iface.get('flags') is not None
        assert len(iface) == 1


f()
