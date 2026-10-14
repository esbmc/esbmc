class X:
    def __init__(self) -> None:
        self.v: int = 1
    def get(self) -> int:
        return self.v

def f() -> bool:
    try:
        raise ValueError("a")
    except ValueError as X:
        return isinstance(X, ValueError)
    return False
