def put(x: list) -> None:
    x[0] = [5]

m: list = [[1]]
n: list = [[1]]
put(m)
assert m == n
