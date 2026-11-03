def put(x: list) -> None:
    x[0] = ""


m: list = [0]
n: list = [0]
put(m)
assert m == n
