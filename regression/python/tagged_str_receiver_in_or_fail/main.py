import random

flag = random.random() < 0.5
x = "ab" if flag else 3
assert flag or x.startswith("a")
