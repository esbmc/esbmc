def is_foo(a: str) -> bool:
    return a != "foo"


e = is_foo(a="foo")
assert e
