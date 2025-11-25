def foo() -> dict:
  d = {"a": 1, "b": 2, "c": 3} 
  return d

my_dict:dict = foo()
assert (my_dict["a"] == 1)
