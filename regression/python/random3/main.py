import random
def test_random_basic():
 x: float = random.random()
 y: int = random.randint(1, 10)
 z: int = int(x * y) + 1 
 print(x)
 assert y>= 1 and y <= 10
 assert z <= 10
test_random_basic()
