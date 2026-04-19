from typing import List

x:List[int] = [1,2,3]
assert x[0] == 1
assert x[1] == 2
assert x[2] == 3

assert len(x) == 3

def sum_while(n: List[int]) -> int:
    l = len(n)
    i = 0
    s = 0
    while i < l:
        s += n[i]
        i += 1
    return s

assert sum_while([1, 2, 3, 4]) == 10