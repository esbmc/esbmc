class Adder:
    def __init__(self, val: int) -> None:
        self.val: int = val
    def __enter__(self) -> int:
        return self.val
    def __exit__(self, exc_type: int, exc_val: int, exc_tb: int) -> bool:
        return False

x: int = 0
y: int = 0
with Adder(3) as a, Adder(7) as b:
    x = a
    y = b

assert x == 3
assert y == 7
