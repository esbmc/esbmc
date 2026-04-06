class MyNum:
    val: int

    def __init__(self, v: int) -> None:
        self.val = v

    def __int__(self) -> int:
        return self.val

    def negate(self) -> "MyNum":
        return MyNum(0 - self.val)

x: MyNum = MyNum(42)
result: int = int(x.negate())
assert result == -42
