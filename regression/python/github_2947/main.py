def foo() -> tuple[int, int]:
    return (0, 0)

assert foo() == (0, 0)

def get_coords() -> tuple[int, int]:
    return (10, 20)

assert get_coords() == (10, 20)

