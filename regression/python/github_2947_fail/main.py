def foo() -> tuple[int, int]:
    return (0, 0)

if nondet_bool():
    assert foo() == (1, 0)

def get_coords() -> tuple[int, int]:
    return (10, 20)

if nondet_bool():
    assert get_coords() == (11, 20)

