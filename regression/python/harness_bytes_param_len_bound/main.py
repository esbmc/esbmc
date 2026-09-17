# len() on a bytes parameter resolves to the harness backing array's size.
def f(data: bytes) -> int:
    assert len(data) <= 4
    return 0
