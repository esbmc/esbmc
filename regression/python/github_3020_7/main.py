class Calculator:
    def add(self, x: int, y: int) -> int:
        return x + y

calc = Calculator()
result = calc.add(3, 7)
assert result == 10
