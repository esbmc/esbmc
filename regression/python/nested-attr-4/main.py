# Test case 4: Nested attribute as parameter and chained calls
class Calculator:

    def add(self, x: int, y: int) -> int:
        return x + y

    def multiply(self, x: int, y: int) -> int:
        return x * y


class MathService:

    def __init__(self) -> None:
        self.calc: Calculator = Calculator()

    def compute(self, a: int, b: int) -> int:
        # Pass nested attribute method result as parameter
        sum_result: int = self.calc.add(a, b)
        return self.calc.multiply(sum_result, 2)

    def chain_calls(self, x: int) -> int:
        # Chain nested attribute calls
        result = self.calc.multiply(self.calc.add(x, 1), 2)
        return result


service = MathService()
result1: int = service.compute(3, 4)  # (3+4)*2 = 14
result2: int = service.chain_calls(5)  # (5+1)*2 = 12
assert result1 == 14
assert result2 == 12
