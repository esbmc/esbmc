import random as rand

def div1(cond: int, x: int) -> int:
    if (not cond):
        return 42 // x
    else:
       return x // 10

cond:int = rand.random()
x:int = rand.random()

assert div1(cond, x) != 1
