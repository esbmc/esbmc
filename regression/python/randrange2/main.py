import random

result: int = random.randrange(10)
assert result >= 0 and result < 10

result2: int = random.randrange(5)
assert result2 >= 0 and result2 < 5
