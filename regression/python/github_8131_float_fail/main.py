def put(x: list) -> None:
    x[0] = 1.5


m: list = [1]
n: list = [1]
put(m)
assert m == n
