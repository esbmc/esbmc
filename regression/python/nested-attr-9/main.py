# Test case 9: Nested attribute in complex expression
class Math:
    def square(self, x: int) -> int:
        return x * x

    def cube(self, x: int) -> int:
        return x * x * x

class Calculator:
    def __init__(self) -> None:
        self.math: Math = Math()

class Service:
    def __init__(self) -> None:
        self.calc: Calculator = Calculator()

    def compute(self, x: int) -> int:
        # Complex expression with multiple nested attribute calls
        squared = self.calc.math.square(x)
        cubed = self.calc.math.cube(x)
        result = squared + cubed
        return result

service = Service()
result: int = service.compute(3)  # 3^2 + 3^3 = 9 + 27 = 36
assert result == 36
