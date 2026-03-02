def not_called(s: str) -> None:
    b: bytes = s.encode("utf-8")
    assert len(b) > 0


def foo(i: int) -> bool:
    return i > 5


assert not foo(0)
