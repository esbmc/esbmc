def put(x: list) -> None:
    x[0] = "abc"


m: list = [0]
n: list = ["abc"]
put(m)
assert m == n
assert m != [0]
