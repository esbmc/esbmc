# 10434 control: an int-or-None name that holds None is not an int.
import random

s = random.randint(0, 9) if random.random() < 0.5 else None
assert isinstance(s, int) == (s is not None)
