def foo(s: str | None = None, t: str | None = None) -> None:
    d: list[str] = []
    if s is None:
        d.append(s)
    if t is not None:
        d.append(t)
    
    l = len(d)
    assert l == 1

foo(s="foo")
