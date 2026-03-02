def greet(name: str) -> str:
    return name


def add(a: int, b: int) -> int:
    return a + b


def process_data(value: int, factor: float, label: str) -> float:
    return value * factor


result1 = add(5, 10)  # Correct - both arguments are int
assert result1 == 15

result3 = add(5, "10")  # TypeError: second argument is str, expected int

result4: str = greet(42)  # TypeError: argument is int, expected str

result5 = process_data("wrong", 2.5, "label")  # TypeError: first argument is str, expected int
