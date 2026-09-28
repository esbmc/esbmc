def is_foo(a: str) -> bool:
    return a == "foo"


hit = is_foo(a="foo")
miss = is_foo(a="bar")
assert hit
assert not miss
