from typing import Optional


def foo(x: int, y: Optional[str] = None, z: Optional[int] = None) -> int:
    assert y is not None or z is not None
    return 42


foo(1)
