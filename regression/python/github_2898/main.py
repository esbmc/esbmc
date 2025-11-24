def check(s: str) -> None:
    l: list[str] = ['foo', 'bar', 'baz']
    assert s in l

check('foo')
