import random

x = random.randint(0, 10)

my_dict = {"a": x, "b": 2, "c": 3}
assert (my_dict["a"] == x)
del my_dict["b"]
del my_dict["c"]
assert "a" in my_dict
