def check(s: str) -> None:
    l: list[str] = ['foo', 'bar', 'baz']
    assert not s in l


check('foo')
