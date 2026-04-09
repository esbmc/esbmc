def fibonacci(n: int) -> int:
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 1
    # Missing return for n > 2
    fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(5)
assert result == 5
