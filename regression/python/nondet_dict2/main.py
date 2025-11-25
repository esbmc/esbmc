def foo(d: dict) -> dict:
  d["a"] = 2
  return d

d = {"a": 1, "b": 2, "c": 3}

my_dict:dict = foo(d)

assert (my_dict["a"] == 2)
