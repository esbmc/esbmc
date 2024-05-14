def factorial(n:int) -> int:
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

result:int = factorial(5)
assert(result == 120)

result = factorial(4)
assert(result == 20) # result == 24