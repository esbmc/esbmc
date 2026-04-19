import random

x = random.randint(0, 100)
y = random.randint(0, 100)
z = random.randint(0, 100)

def foo() -> dict:
  d = {"a": x, "b": y, "c": z} 
  return d

my_dict:dict = foo()
assert (my_dict["a"] == x)
assert (my_dict["b"] == y)
assert (my_dict["c"] == z)
