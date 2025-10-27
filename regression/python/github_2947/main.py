def foo() -> tuple[int, int]:
    return (0, 0)

foo()

def get_coords() -> tuple[int, int]:
    return (10, 20)

get_coords()

# Mixed types with annotation
def get_info() -> tuple[str, int, float]:
    return ("Alice", 30, 5.9)

