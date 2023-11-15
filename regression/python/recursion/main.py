def fib(n:int) -> int:
  if n <= 0:
    return 0
  elif n == 1:
    return 1
  else:
    return fib(n - 1) + fib(n - 2)

n:int = 6
result:int = fib(n)