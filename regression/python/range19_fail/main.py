import random

result = 0

def factorial (n):
  if n < 0:
    assert(0)
  
  result = 1
  
  for i in range (1, n + 1):
    result *= i
  
  return result

x = random.randint(1, 16383)
assert factorial (x) < 0
