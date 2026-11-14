import random

flag = random.random() < 0.5
x = "ab" if flag else 3
assert not flag or x.startswith("a")
assert flag or x == 3
