# The backing array's length is nondeterministic, not a fixed 0.
def f(data: bytes) -> int:
    assert len(data) == 0
    return 0
