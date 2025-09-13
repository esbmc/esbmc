def safe_div(a: int, b: int) -> float:
    if b == 0:
        raise ZeroDivisionError("Division by zero!")
    return a / b

result = 10
try:
    result = safe_div(5, 0)
except ZeroDivisionError as e:
    print(e)

assert result == 10
