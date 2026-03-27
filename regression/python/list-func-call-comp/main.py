def foo() -> list[int]:
    return [1, 2, 3]


y = [1, 2, 3]
assert foo() == y
