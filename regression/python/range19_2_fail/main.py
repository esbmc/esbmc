import random

result:int = 0

def factorial (n: int) -> int:
  if n < 0:
    assert(0)
  
  result = 1
  
  for i in range (1, n + 1):
    result *= i
  
  return result

x:int = 4
assert factorial (x) == 25
