class Ctx:
    def __enter__(self) -> int:
        return 42
    def __exit__(self, exc_type: int, exc_val: int, exc_tb: int) -> bool:
        return False

x: int = 0
with Ctx() as v:
    x = v

assert x == 0
